"""Turso 金句庫的每週備份：整庫複製成 SQLite、gzip 後放 Drive 的 `_psr/backups/`，保留最近幾份。

不放 repo 或 Actions artifact：內容有版權，而這是公開 repo。
"""

from __future__ import annotations

import re
from datetime import date

FOLDER = "backups"
_NAME = re.compile(r"^bob-\d{4}-\d{2}-\d{2}\.sqlite\.gz$")


def name_for(day: date) -> str:
    return f"bob-{day:%Y-%m-%d}.sqlite.gz"


def to_trash(names: list[str], keep: int) -> list[str]:
    """最舊的那些備份（只看本程式產生的檔名），讓資料夾裡留下最新 `keep` 份。"""
    ours = sorted(n for n in names if _NAME.match(n))
    return ours[:max(len(ours) - keep, 0)]
