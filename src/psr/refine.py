from psr.models import Word, Cue
from psr.text import display_width


def merge_short_cues(
    cues: list[Cue],
    max_width: float = 36.0,
    max_gap: float = 0.4,
    max_s: float = 12.0,
    soft_min: float = 16.0,
    hard_max: float = 46.0,
) -> list[Cue]:
    """把過碎的字幕往回合併，直到接近行寬上限為止。

    存在理由：潤稿的 8% 編輯距離不變量只驗證「內容」，完全不管「切法」——
    LLM 回傳 ["大","家","好"] 能完美通過檢查，因為接起來的字串一模一樣。
    87 分鐘實片實測，模型產出的 2,425 條字幕裡有 1,338 條（55%）不到兩個
    全形字，最短的是單個英文字母 'I'、'A'、't'。

    與其在 prompt 裡跟模型爭執每行要幾個字（實測遵守度極不穩定），不如接受
    它給的語意邊界、再用確定性的規則合併。**只合併、永不切分**，所以不會
    製造 LLM 沒同意過的斷點，也不會動到時間軸——合併後的區間就是兩端的聯集。

    合併以**標點為界**，而不是湊到字數上限。潤稿已經把標點加上去了，所以
    標點就是天然的斷點——目標是「一條字幕 = 一個讀得完整的語句」。

    兩種停止條件：

      硬停：句尾標點（。！？）。一句話講完就換一條字幕。
      軟停：逗號頓號，且累積寬度已達 soft_min。短句因此能保持完整，
            長句則在逗號處切成長度合理的區塊，而不是硬塞進一條再折行。

    另外三個上限同時成立才會合併：相鄰間隔 ≤ max_gap（跨越長靜音不合併）、
    合併後行寬 ≤ max_width、合併後時長 ≤ max_s。
    """
    if not cues:
        return []
    out = [cues[0]]
    for cue in cues[1:]:
        prev = out[-1]
        joined = prev.text + cue.text
        # 寬度上限卡住時就地停下，會把斷點留在詞中間（實測切出「…一個聊天」／
        # 「的界面…」）或留下「那」「你就會」這種孤兒字幕。所以尚未走到任何
        # 標點時放寬到 hard_max，讓它有機會走到自然斷點——一條稍寬的字幕，
        # 遠比一個斷在詞中間的字幕好讀。
        width_ok = display_width(joined) <= max_width or (
            not _at_punctuation(prev.text) and display_width(joined) <= hard_max
        )
        # 碎片（任一側去空白後不足兩個全形字寬）一律合併，不受間隔限制。
        # 一條只有 'I' 或 'ys' 的字幕永遠是錯的——閃一下、讀不到、還佔著
        # 螢幕時間。實測潤稿模型在少數窗口會退回逐字元切分，產出大量這種
        # 碎片，讓它們因為跨越靜音而留在原地只是把模型的失誤原樣呈現。
        fragment = min(display_width(prev.text.strip()),
                       display_width(cue.text.strip())) < 2
        if (
            not _should_stop(prev.text, soft_min)
            and (fragment or cue.start - prev.end <= max_gap)
            and (width_ok or fragment)
            and cue.end - prev.start <= max_s
        ):
            out[-1] = Cue(index=prev.index, start=prev.start, end=cue.end, text=joined)
        else:
            out.append(cue)
    return _reindex(out)


FRAGMENT_WIDTH = 2.0


def absorb_fragments(cues: list[Cue], min_width: float = FRAGMENT_WIDTH) -> list[Cue]:
    """把碎片字幕併進鄰居，直到一條都不剩。

    merge_short_cues 已經會合併碎片，但它同時受標點停止條件約束——碎片剛好
    跟在句號後面時就不會被吸收。實測 87 分鐘影片仍留下 184 條單字元字幕。

    這一道獨立的收尾不看標點也不看間隔，只問一件事：這條字幕**讀得到嗎**。
    一條寬度不到兩個全形字的字幕在螢幕上就是閃一下，讀不到卻佔著時間，
    無論它前面是句號還是逗號都一樣錯。

    併入哪一側取決於時間距離：黏給比較近的鄰居，這樣時間軸的失真最小。
    """
    if not cues:
        return []
    out: list[Cue] = []
    for cue in cues:
        if out and display_width(cue.text.strip()) < min_width:
            prev = out[-1]
            out[-1] = Cue(prev.index, prev.start, cue.end, prev.text + cue.text)
            continue
        if out and display_width(out[-1].text.strip()) < min_width:
            prev = out.pop()
            cue = Cue(prev.index, prev.start, cue.end, prev.text + cue.text)
        out.append(cue)
    return _reindex(out)


_SENTENCE_END = "。！？!?…"
_LEADING_JUNK = "，,、；;：:。！？!?"
_WRAP_PREFERRED = "，,、；;：:"


def strip_leading_punctuation(cues: list[Cue]) -> list[Cue]:
    """刪掉字幕開頭的孤兒標點。

    講者在句中長停頓時，潤稿的斷點會落在逗號前，而合併因為間隔太大不會跨越
    那個停頓，於是下一條字幕以「，」開頭。實測：「大家好，我是林協廷醫師」／
    「，今天這堂課是……」。
    """
    stripped = [
        Cue(index=c.index, start=c.start, end=c.end, text=c.text.lstrip(_LEADING_JUNK).lstrip())
        for c in cues
    ]
    # 整條只有標點的字幕，剝完就變成空的。空白字幕會佔著螢幕時間卻什麼都
    # 不顯示，而且**其他所有規則都抓不到它**：時長合法、不重疊、行寬 0、
    # 閱讀速度 0。實測第四次 CI 執行產出 6 條這樣的字幕——先前在
    # _split_one 加的守衛只堵住了其中一個產生途徑，這裡是另一個。
    return _reindex([c for c in stripped if c.text.strip()])


def wrap_lines(cues: list[Cue], max_width: float = 18.0) -> list[Cue]:
    """把過寬的字幕折成兩行顯示（插入換行字元），**不改動時間軸**。

    這是「語句完整」與「單行可讀」之間的解法：合併階段以句尾標點為界、讓
    一條字幕裝下一個完整語句，寬度上限放寬到 36；顯示時再折行，而不是在
    合併階段為了湊 20 個字把句子切在詞中間（實測會切出「聊」／「天」、
    「樣」／「子」這種斷點）。

    折行點優先選逗號頓號之後——那是語句內的自然停頓；沒有的話取中點附近。
    """
    out: list[Cue] = []
    for cue in cues:
        text = cue.text
        if display_width(text) <= max_width or "\n" in text:
            out.append(cue)
            continue
        out.append(Cue(index=cue.index, start=cue.start, end=cue.end, text=_wrap(text)))
    return out


_WRAP_TARGET = 20.0


def _wrap(text: str) -> str:
    half = display_width(text) / 2
    best, best_dist = None, None
    acc = 0.0
    for i, ch in enumerate(text):
        acc += display_width(ch)
        if ch in _WRAP_PREFERRED:
            dist = abs(acc - half)
            if best_dist is None or dist < best_dist:
                best, best_dist = i + 1, dist
    if best is None:                      # 沒有標點可用就取中點
        acc = 0.0
        for i, ch in enumerate(text):
            acc += display_width(ch)
            if acc >= half:
                best = i + 1
                break
    if not best or best >= len(text):
        return text
    left, right = text[:best], text[best:]
    # 對折一次還是太寬時再折一次（46 字對折是 23 字，仍超過單行 20 的上限）。
    # 上限三行——再多就不是字幕而是段落了。
    if display_width(right) > _WRAP_TARGET and "\n" not in right:
        right = _wrap(right)
    return left + "\n" + right


def _at_punctuation(text: str) -> bool:
    """這段文字是不是剛好停在某個標點上？不是的話，斷在這裡就是斷在詞中間。"""
    return bool(text) and text.rstrip()[-1:] in (_SENTENCE_END + _WRAP_PREFERRED)


def _should_stop(text: str, soft_min: float) -> bool:
    """這段文字是否該就此結束、不再往後合併？

    句尾標點一律停。逗號頓號只有在累積長度已經夠讀時才停——太早就在逗號斷，
    會切出「那我想，」這種讀不完整的半句。
    """
    tail = text.rstrip()[-1:]
    if not tail:
        return False
    if tail in _SENTENCE_END:
        return True
    return tail in _WRAP_PREFERRED and display_width(text) >= soft_min


def enforce_duration(
    cues: list[Cue],
    words: list[Word],
    min_s: float = 0.5,
    max_s: float = 12.0,
    audio_duration: float | None = None,
) -> list[Cue]:
    """修正字幕時長：太長的在字間最大間隔處遞迴切開；太短的往後延或向鄰居
    借時間。絕不製造重疊。

    audio_duration 給定時，最後一條字幕不會被延長超過音訊結尾——否則
    validate 的邊界檢查會判它違規。省略時不設上限（最後一條照常延到 min_s）。
    """
    split_cues = _split_long_cues(cues, words, max_s)
    return _extend_short_cues(split_cues, min_s, audio_duration)


def _words_in_cue(cue: Cue, words: list[Word]) -> list[Word]:
    return [w for w in words if w.start >= cue.start and w.end <= cue.end]


def _split_long_cues(cues: list[Cue], words: list[Word], max_s: float) -> list[Cue]:
    result: list[Cue] = []
    for cue in cues:
        result.extend(_split_one(cue, words, max_s))
    return _reindex(result)


def _split_one(cue: Cue, words: list[Word], max_s: float) -> list[Cue]:
    duration = cue.end - cue.start
    if duration <= max_s:
        return [cue]

    cue_words = _words_in_cue(cue, words)
    if len(cue_words) < 2:
        # 沒有字間間隔可切，只能保留原樣（不製造無中生有的斷點）
        return [cue]
    if len(cue.text.strip()) < 2:
        # 去掉空白後不足兩個字元，切開必然有一半沒有內容。寧可留一條過長的
        # 字幕，也不要產生一條佔著螢幕時間卻沒有字的空白字幕。
        #
        # 這裡用 strip() 而非 len(cue.text)：實測 'I ' 長度是 2 會通過原本的
        # 檢查，然後被對半切成 'I' 與 ' '，後者就是一條空白字幕。
        return [cue]

    # 找字間最大間隔作為切點
    best_gap = -1.0
    split_at = None
    for i in range(len(cue_words) - 1):
        gap = cue_words[i + 1].start - cue_words[i].end
        if gap > best_gap:
            best_gap = gap
            split_at = i

    left_words = cue_words[: split_at + 1]
    right_words = cue_words[split_at + 1 :]

    # 切點的字元位置用 ASR 左半的字元數估計，然後**箝制在兩側都非空的範圍內**。
    #
    # 這個估計本身是不精確的：那是 ASR 原始逐字稿的字元數，而 cue.text 是潤稿
    # 後的文字，兩者字元不保證對應——潤稿會改字、加標點、刪贅字，那正是它的
    # 目的。但潤稿多半只微調長度，所以估計在常態下是準的，比用時間比例去推
    # 字元位置更好（講者停頓時時間比例會嚴重偏離字元比例）。
    #
    # 箝制不是美化，是防災。潤稿若把句子大幅縮短，未箝制的 split_char 會超出
    # 文字長度，cue.text[:split_char] 回傳整串、cue.text[split_char:] 回傳空
    # 字串，產出一條佔著數秒螢幕時間卻沒有字的空白字幕——而且不拋任何錯誤，
    # 時長合法、不重疊、行寬 0、閱讀速度 0，所有其他規則都抓不到它。
    split_time = left_words[-1].end
    split_char = sum(len(w.text) for w in left_words)
    split_char = max(1, min(len(cue.text) - 1, split_char))

    left_text = cue.text[:split_char]
    right_text = cue.text[split_char:]
    if (display_width(left_text.strip()) < FRAGMENT_WIDTH
            or display_width(right_text.strip()) < FRAGMENT_WIDTH):
        # 箝制保證了兩半非空「字串」，但沒保證兩半**讀得到**。'I ' 切成
        # 'I' 與 ' ' 是極端案例，「所以當」切成「所以」與「當」則是常態——
        # 兩者都是佔著螢幕時間卻讀不到的碎片。寧可留一條過長的字幕。
        #
        # 這道檢查必須在切分處，不能只靠事後的 absorb_fragments：那一步跑在
        # enforce_duration 之前，切分產生的新碎片它看不到。
        return [cue]

    # 保留 cue 原本的外側邊界，只在中間切開。用 left_words[0].start 當左半起點
    # 會在 cue 起始於某個 word 內部時丟失原本的起始時間。
    left = Cue(index=cue.index, start=cue.start, end=split_time, text=left_text)
    right = Cue(index=cue.index, start=right_words[0].start, end=cue.end, text=right_text)

    return _split_one(left, words, max_s) + _split_one(right, words, max_s)


_EPS = 1e-9


def _extend_short_cues(
    cues: list[Cue], min_s: float, audio_duration: float | None = None
) -> list[Cue]:
    """把過短的字幕延長到 min_s。

    這個函式會收到兩種形狀完全不同的輸入，兩種都必須處理：

      - segment.raw_segment 的輸出「有空檔」→ 直接往空檔延伸即可
      - align 的輸出「首尾相接」（cue[i].end == cue[i+1].start）→ 空檔為零，
        只能向鄰居借時間，也就是推移兩者共用的邊界，且只借到鄰居本身仍保有
        min_s 為止

    只實作前者的話，align 輸出中任何過短的字幕都會原封不動通過，接著被
    validate 判為違規、導致整個窗口降級——延長功能形同虛設。
    """
    result = list(cues)
    for i, cue in enumerate(result):
        start, end = cue.start, cue.end
        if end - start >= min_s:
            continue

        # 1) 先吃掉與下一條之間的空檔（若有）。用 max 避免在異常輸入下反而縮短。
        if i + 1 < len(result):
            limit = result[i + 1].start
        else:
            # 最後一條沒有下一條擋著，但不能延過音訊結尾。
            limit = start + min_s if audio_duration is None else min(start + min_s, audio_duration)
        end = min(start + min_s, max(end, limit))

        # 2) 仍不足就向右鄰借，推移共用邊界。借完右鄰仍有 min_s，
        #    所以之後輪到它時不會被判定為過短，不會連鎖。
        if start + min_s - end > _EPS and i + 1 < len(result):
            nxt = result[i + 1]
            borrow = min(start + min_s - end, max(0.0, (nxt.end - nxt.start) - min_s))
            if borrow > 0:
                end += borrow
                result[i + 1] = Cue(index=nxt.index, start=end, end=nxt.end, text=nxt.text)

        # 3) 還不足就向左鄰借。左鄰已處理過，同樣只借到它仍保有 min_s。
        if start + min_s - end > _EPS and i > 0:
            prv = result[i - 1]
            borrow = min(start + min_s - end, max(0.0, (prv.end - prv.start) - min_s))
            if borrow > 0:
                start -= borrow
                result[i - 1] = Cue(index=prv.index, start=prv.start, end=start, text=prv.text)

        # 兩邊都借不到時就維持原樣：這代表整段真的沒有空間，
        # 交給 validate 攔下並降級，比硬造出重疊誠實。
        result[i] = Cue(index=cue.index, start=start, end=end, text=cue.text)
    return result


def _reindex(cues: list[Cue]) -> list[Cue]:
    return [Cue(index=i + 1, start=c.start, end=c.end, text=c.text) for i, c in enumerate(cues)]
