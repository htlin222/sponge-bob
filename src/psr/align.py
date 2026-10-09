import bisect
import difflib

from psr.models import Cue, Word
from psr.text import normalize, to_traditional


def word_stream(words: list[Word]) -> tuple[str, list[int]]:
    """把 word 清單串接成一條字元流，並回傳每個字元對應的 word index。
    這是原始側（ASR 逐字稿）的字元流。"""
    text_parts: list[str] = []
    char_to_word: list[int] = []
    for word_idx, word in enumerate(words):
        for ch in word.text:
            text_parts.append(ch)
            char_to_word.append(word_idx)
    return "".join(text_parts), char_to_word


def line_stream(lines: list[str]) -> tuple[str, list[int], list[int]]:
    """把字幕行清單串接成一條字元流。這是潤稿側的字元流。

    回傳 (串接後文字, char_index -> line_index 映射, 每行起始的 char offset 清單)。
    boundaries 的第一個元素永遠是 0（第一行從頭開始）。
    """
    text_parts: list[str] = []
    char_to_line: list[int] = []
    boundaries: list[int] = []
    offset = 0
    for line_idx, line in enumerate(lines):
        boundaries.append(offset)
        for ch in line:
            text_parts.append(ch)
            char_to_line.append(line_idx)
        offset += len(line)
    return "".join(text_parts), char_to_line, boundaries


def map_boundaries(norm_orig: str, norm_poly: str, boundaries: list[int]) -> list[int | None]:
    """把潤稿正規化字串上的斷點位置，映射回原始正規化字串上的位置。

    autojunk=False 是強制的：預設的 autojunk 啟發式會把「長度 >=200 的序列中
    出現超過 1% 的元素」當雜訊丟掉。中文的「的」「是」「我」「這」這類高頻字
    在長逐字稿裡幾乎必中，一旦被當雜訊丟掉，equal 區塊會靜默地變少變爛——
    不會拋例外，只會給出錯誤的時間軸。
    """
    matcher = difflib.SequenceMatcher(None, norm_orig, norm_poly, autojunk=False)
    opcodes = matcher.get_opcodes()
    return [_map_one_boundary(opcodes, b) for b in boundaries]


def _map_one_boundary(opcodes, b: int) -> int | None:
    equal_blocks = [
        (i1, i2, j1, j2) for tag, i1, i2, j1, j2 in opcodes if tag == "equal" and j1 < j2
    ]

    # 精確落在某個 equal 區塊內：線性映射
    for i1, i2, j1, j2 in equal_blocks:
        if j1 <= b < j2:
            return i1 + (b - j1)

    if not equal_blocks:
        return None

    # 吸附到最近的 equal 區塊邊界。用整數距離比較，平手時取 edge_j 較小
    # （較早出現）的那個候選——這是決定性 tie-break，不依賴浮點數或字典順序。
    best_edge_i: int | None = None
    best_edge_j: int | None = None
    best_dist: int | None = None
    for i1, i2, j1, j2 in equal_blocks:
        for edge_j, edge_i in ((j1, i1), (j2, i2)):
            dist = abs(edge_j - b)
            if (
                best_dist is None
                or dist < best_dist
                or (dist == best_dist and edge_j < best_edge_j)
            ):
                best_dist = dist
                best_edge_j = edge_j
                best_edge_i = edge_i
    return best_edge_i


def has_anchor(opcodes, orig_lo: int, orig_hi: int, min_len: int = 4) -> bool:
    """檢查 [orig_lo, orig_hi) 這段原始字元區間內，是否至少有一個夠長的
    equal 區塊與它重疊。這是防止「斷點映射得到值」但「這段內容其實已經
    面目全非」的防線——map_boundaries 只保證吸附得到一個位置，不保證那個
    位置附近的內容真的還能在原文裡找到對應。

    要求的長度會**隨跨度縮小**：跨度只有 2 個字元時，不可能容納 4 個字元的
    錨點，硬要求 4 就是要求一件做不到的事。實測後果嚴重——一份逐字完全相同
    的潤稿，只因為其中一行是「好的」這種兩字短句，整個窗口就被判定失敗而
    降級。87 分鐘實片有 7/14 個窗口是這樣掉的。
    """
    required = min(min_len, orig_hi - orig_lo)
    if required <= 0:
        return False
    min_len = required
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            continue
        overlap_lo = max(i1, orig_lo)
        overlap_hi = min(i2, orig_hi)
        if overlap_hi - overlap_lo >= min_len:
            return True
    return False


def _char_offsets_in_word(words: list[Word]) -> tuple[list[int], list[int]]:
    """對原始字元流的每個位置，回傳它在所屬 word 內的位移與該 word 的總字元數。
    這是 word 內線性內插的輸入。"""
    offsets: list[int] = []
    lengths: list[int] = []
    for word in words:
        n = len(word.text)
        for k in range(n):
            offsets.append(k)
            lengths.append(n)
    return offsets, lengths


def _boundary_times(
    norm_pos: int,
    orig_norm_map: list[int],
    char_to_word: list[int],
    char_offsets: list[int],
    word_lengths: list[int],
    words: list[Word],
) -> tuple[float, float]:
    """把一個斷點換算成 (前一行的結束時間, 後一行的開始時間)。

    一個斷點要兩個時間而不是一個，因為它同時是前一行的結尾與後一行的開頭，
    而這兩者**不一定相同**——中間可能隔著靜音。兩種情況必須分開處理：

      落在 word 內部（offset > 0）：
        兩者相同，取線性內插值。此時前後兩行在時間上首尾相接。
        必須如此，否則前一行取該 word 的 end、後一行取同一個 word 的 start，
        會產生時間軸重疊。中文的 Whisper word 常是多字詞、重切字幕行本來就會
        切進 word 內部，所以這是常態而非邊緣案例。

      落在 word 之間（offset == 0）：
        前者取前一個 word 的 end，後者取這個 word 的 start，中間的靜音**保留**。
        螢幕錄製的講者常在操作畫面時停頓十幾秒，把那段靜音吸進前一條字幕，
        會讓一句話在螢幕上僵住十幾秒。靜音時本來就不該有字幕。

    因為 words 的時間單調不重疊，prev.end <= word.start 恆成立，
    所以無論走哪一支都不可能產生重疊。
    """
    if norm_pos >= len(orig_norm_map):
        return words[-1].end, words[-1].end

    orig_char = orig_norm_map[norm_pos]
    word_idx = char_to_word[orig_char]
    word = words[word_idx]
    length = word_lengths[orig_char]
    offset = char_offsets[orig_char]

    if offset > 0 and length > 0:
        instant = word.start + (offset / length) * (word.end - word.start)
        return instant, instant

    if word_idx == 0:
        return word.start, word.start
    return words[word_idx - 1].end, word.start


def align(words: list[Word], lines: list[str]) -> list[Cue] | None:
    """把 LLM 潤稿後的字幕行貼回 ASR 的 word 時間軸。

    LLM 潤稿時完全看不到任何時間碼（設計文件 §6），時間軸永遠只從
    words 的 start/end 推導，杜絕潤稿把時間戳一起「潤」壞的可能。

    任一行找不到足夠錨點時，整個窗口降級為 None（呼叫端應改用
    segment.raw_segment）——只降級單行會在前後製造時間軸縫隙，
    縫合邏輯本身就是新的 bug 來源，所以採取「一行壞、全窗降級」的保守策略。

    時間軸由每個斷點推導出一對 (前一行結束, 後一行開始) 時間（見
    _boundary_times）。斷點落在 word 內部時兩者相同、字幕首尾相接；落在
    word 之間時保留中間的靜音。因此 cue[i].end <= cue[i+1].start 恆成立，
    不重疊是建構上的保證，而不是事後檢查出來的性質。
    """
    if not words or not lines:
        return None

    orig_text, char_to_word = word_stream(words)
    poly_text, char_to_line, boundaries = line_stream(lines)

    norm_orig, orig_norm_map = normalize(orig_text)
    norm_poly, poly_norm_map = normalize(poly_text)

    # 把「poly 原始字元流上的行起點」換算成「poly 正規化字元流上的位置」。
    # poly_norm_map 是 normalized_pos -> raw_pos（非遞減），bisect_left 找到
    # 第一個 raw_pos >= 目標值的 normalized 位置——也就是「原始位置在正規化
    # 後最靠前但不早於它」的落點，剛好處理掉邊界字元被正規化丟棄的情況。
    boundaries_with_end = boundaries + [len(poly_text)]
    norm_boundaries = [
        bisect.bisect_left(poly_norm_map, raw_b) for raw_b in boundaries_with_end
    ]

    # opcodes 只算一次，map_boundaries 內部原本會用同樣的輸入再算一遍。
    # 結果一致（SequenceMatcher 是決定性的），但長逐字稿上是白費的成本。
    matcher = difflib.SequenceMatcher(None, norm_orig, norm_poly, autojunk=False)
    opcodes = matcher.get_opcodes()
    mapped = [_map_one_boundary(opcodes, b) for b in norm_boundaries]
    if any(m is None for m in mapped):
        return None

    char_offsets, word_lengths = _char_offsets_in_word(words)
    # 每個斷點兩個時間：(前一行結束, 後一行開始)。落在靜音處時兩者不同。
    times = [
        _boundary_times(m, orig_norm_map, char_to_word, char_offsets, word_lengths, words)
        for m in mapped
    ]

    cues: list[Cue] = []
    for line_idx, line in enumerate(lines):
        lo = mapped[line_idx]
        hi = mapped[line_idx + 1]
        if lo >= hi:
            return None
        if not has_anchor(opcodes, lo, hi):
            return None

        start = times[line_idx][1]        # 這一行從它前面那個斷點的「後」開始
        end = times[line_idx + 1][0]      # 到它後面那個斷點的「前」為止
        if not start < end:
            return None

        # 繁化在這裡做而不是交給潤稿 prompt：查表轉換是確定性的，LLM 不是。
        # 對已是繁體的文字是恆等操作，所以即使潤稿已經輸出繁體也無害。
        cues.append(Cue(index=line_idx + 1, start=start, end=end, text=to_traditional(line)))

    return cues
