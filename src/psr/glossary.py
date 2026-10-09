"""術語表載入與衍生格式。

glossary.yml 是一份餵兩處的來源（見該檔案開頭註解）：
  - Whisper 的 prompt 參數（上限 224 tokens）→ whisper_prompt()
  - 潤稿 LLM 的 system prompt（可用完整表）    → polish_hint()

兩者都必須是「檔案內容的純函式」：同一份 glossary.yml 永遠產生同樣的
prompt 字串，不做任何依執行環境變動的排序或篩選。這是刻意的約束——
polish.py 用 glossary.content_hash() 當 pipeline 的 stage key 的一部分，
一旦 prompt 內容會隨機或依執行時的「相關性排序」變動，快取失效判斷
就會失準（明明術語表沒變，key 卻變了；或術語表變了，key 卻沒變）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

# Whisper prompt 參數的官方上限。
WHISPER_PROMPT_TOKEN_LIMIT = 224

# 這裡沒有 tiktoken 依賴（專案目前不需要它做其他事），所以用保守
# （偏高）的字元換算估計 token 數，而不是精確計算。偏高是刻意的方向：
# 「填到接近上限就停」這個貪婪演算法，寧可少填幾個詞、也不能因為低估
# 而讓組出來的 prompt 實際送到 Whisper API 時真的超過 224 tokens 被拒。
#   - 寬字元（CJK／全形，含中文術語與人名）：每字算 2 tokens。
#     BPE tokenizer 對中文常見字元的實際切法多在 1~3 tokens/字元之間，
#     取 2 作為偏保守的中位估計。
#   - 其餘字元（英文、數字、符號）：每 3 字算 1 token，對應「英文一個
#     token 約 3~4 字元」的常見經驗值，同樣取偏保守的一端。
_WIDE_TOKENS_PER_CHAR = 2.0
_NARROW_CHARS_PER_TOKEN = 3.0


def _is_wide(ch: str) -> bool:
    import unicodedata

    return unicodedata.east_asian_width(ch) in ("W", "F")


def _estimate_tokens(s: str) -> float:
    total = 0.0
    for ch in s:
        if _is_wide(ch):
            total += _WIDE_TOKENS_PER_CHAR
        else:
            total += 1.0 / _NARROW_CHARS_PER_TOKEN
    return total


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """術語表的一列。"""

    correct: str
    wrong: tuple[str, ...]
    whisper_hint: bool
    note: str | None


@dataclass(frozen=True, slots=True)
class Glossary:
    """已解析的術語表，連同原始檔案位元組（供 content_hash 用）。"""

    entries: tuple[GlossaryEntry, ...]
    raw_bytes: bytes

    def whisper_prompt(self) -> str:
        """組出 Whisper prompt 參數用的字串。

        只取 whisper_hint: true 的條目，依檔案順序、貪婪填充直到接近
        224-token 上限就停。**不排序、不挑「最相關」的詞**——順序一變，
        prompt 內容就變成依執行時狀態決定，content_hash 當 stage key 的
        意義就沒了（見本檔案開頭的說明）。
        """
        terms = [e.correct for e in self.entries if e.whisper_hint]
        picked: list[str] = []
        for term in terms:
            candidate = picked + [term]
            if _estimate_tokens("，".join(candidate)) > WHISPER_PROMPT_TOKEN_LIMIT:
                break
            picked = candidate
        return "，".join(picked)

    def polish_hint(self) -> str:
        """組出潤稿 LLM system prompt 用的緊湊對照表字串。

        格式：`正確←錯誤1/錯誤2；正確←錯誤1/錯誤2；...`。
        只取有 wrong 清單的條目，每條最多列前兩個錯誤轉錄——多列沒有
        邊際效益，只會拉長 system prompt 的 token 成本。
        """
        parts = [
            f'{e.correct}←{"/".join(e.wrong[:2])}'
            for e in self.entries
            if e.wrong
        ]
        return "；".join(parts)

    def content_hash(self) -> str:
        """glossary.yml 原始位元組的 sha256，供 pipeline stage key 使用。"""
        return hashlib.sha256(self.raw_bytes).hexdigest()


def load(path: str | Path) -> Glossary:
    """讀取並解析 glossary.yml。"""
    raw_bytes = Path(path).read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    terms = data.get("terms") or []
    entries = tuple(
        GlossaryEntry(
            correct=t["correct"],
            wrong=tuple(t.get("wrong") or ()),
            whisper_hint=bool(t.get("whisper_hint", False)),
            note=t.get("note"),
        )
        for t in terms
    )
    return Glossary(entries=entries, raw_bytes=raw_bytes)
