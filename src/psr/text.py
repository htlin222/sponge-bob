import unicodedata

from opencc import OpenCC
from zhconv import convert

# s2tw = 簡體 → 繁體，採台灣字形。純查表、無模型、無隨機性，完全確定性——
# 這正是不該把繁化交給 LLM 的理由：那會把一個確定的操作變成不確定的。
#
# 刻意**不用** s2twp（含詞彙轉換）。87 分鐘實片實測，s2twp 只比 s2tw 多改
# 122 字元（0.48%），其中確實有正確的技術詞（界面→介面 ×9、文件→檔案 ×6、
# 數據→資料 ×4），但也有兩類不該發生的改寫：
#
#   打開 → 開啟 ×13   改寫講者原本就自然的台灣口語。逐字稿是「轉錄」，
#                     不是在地化；改寫講者實際說出口的詞是越權。
#   權限 → 許可權 ×7  台灣根本不這樣講，這是微軟式譯法，反而更糟。
#
# 因此字形轉換由這裡確定性地完成，詞彙層的正規化交給 glossary.yml——
# 那張表由使用者維護，控制權在他手上，也不會夾帶整份第三方詞庫的判斷。
_TW = OpenCC("s2tw")


def to_traditional(s: str) -> str:
    """把文字的**字形**統一成繁體中文（台灣）。對已是繁體的文字是恆等操作。

    只轉字形、不轉詞彙：软件 → 軟件（而非軟體）。詞彙層的台灣化屬於
    glossary.yml 的職責，理由見上方註解。
    """
    return _TW.convert(s)


def normalize(s: str) -> tuple[str, list[int]]:
    """正規化字串供比對用，回傳 (正規化後字串, 索引映射)。

    orig_index[i] 是正規化後字串第 i 個字元對應回原始字串 s 的字元索引。

    逐「來源字元」處理，而不是對整個字串一次做正規化，理由在下面每一步都要說明：
    unicodedata.normalize("NFKC", ...) 這類正規化可能把一個字元展開成多個
    字元（例如全形數字、部分相容字元）。如果對整個字串一口氣做 NFKC，
    展開後的字元數就不再跟原始字元數一一對應，索引映射會在這一步就悄悄壞掉，
    之後所有以這張表為基礎的比對——包括 align.py 的斷點映射——全部都是錯的，
    而且不會報錯，只會給出看起來合理但其實對不上時間軸的結果。
    """
    normalized_chars: list[str] = []
    orig_index: list[int] = []

    for i, ch in enumerate(s):
        # 1. NFKC：全形轉半形、相容字元展開。單一字元可能展開成多個。
        nfkc = unicodedata.normalize("NFKC", ch)
        for nfkc_ch in nfkc:
            # 2. 繁簡統一（僅供比對用，不影響最終顯示文字，因為我們只回傳
            #    normalized_chars，原始文字仍保留在 Cue.text 裡）。
            #
            #    注意這裡只統一「字形」，不統一「用詞」：
            #      計算機 / 计算机 → 折疊後相同（字形差異）
            #      軟體   / 软件   → 折疊後不同（用詞差異，软体 vs 软件）
            #    這是刻意的，不是缺陷。zhconv 的詞彙級轉換需要看上下文，
            #    一旦跨字元就無法維持 1:1 的索引映射，與本函式的核心約束衝突。
            #    而且用詞差異本來就該視為真實的內容修改，計入潤稿的編輯距離
            #    預算，而不是被正規化悄悄抹平。
            folded = convert(nfkc_ch, "zh-cn")
            for folded_ch in folded:
                # 3. 拉丁字母 casefold（比 lower() 更嚴格的大小寫無關比較）
                cf = folded_ch.casefold()
                for cf_ch in cf:
                    # 4. 丟棄標點（Unicode 類別開頭 P）、分隔符（開頭 Z）與空白
                    category = unicodedata.category(cf_ch)
                    if category.startswith("P") or category.startswith("Z") or cf_ch.isspace():
                        continue
                    normalized_chars.append(cf_ch)
                    orig_index.append(i)

    return "".join(normalized_chars), orig_index


def display_width(s: str) -> float:
    """計算字串在字幕排版中的顯示寬度。螢幕錄製字幕規定單行 <=20 全形字寬。

    unicodedata.east_asian_width(ch) 回傳 "W"（Wide，如中文）或 "F"
    （Fullwidth，如全形標點）記 1.0；其餘（拉丁字母、數字、半形符號）記 0.5，
    這樣「Python」(6 個半形字元) 只佔 3.0 個全形字寬，符合視覺上的實際佔用。
    """
    width = 0.0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 1.0
        else:
            width += 0.5
    return width
