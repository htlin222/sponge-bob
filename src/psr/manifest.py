"""內容定址 pipeline 的 provenance 記錄與 stage key 計算（設計文件 §3, §5）。

pipeline 不是線性流程，而是一組內容定址的純階段：每個階段的 key =
`hash(階段名 + 階段版本 + 輸入雜湊 + 參數)`。Drive 中已存在且 key 相符的產物
直接重用，絕不重算——這讓 rerun 逐位元相同，resume 免費附贈。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def stage_key(stage: str, version: str, inputs: list[str], params: dict[str, Any]) -> str:
    """算出一個階段的內容定址 key：sha256 hex digest 截斷至 16 碼。

    `params` 一律用 `sort_keys=True` 序列化（遞迴排序所有巢狀 dict 的
    key），所以 dict 的建構順序永遠不會影響 key。`inputs` 是有序 list，
    保留呼叫端給定的順序。
    """
    canonical = json.dumps(
        {"stage": stage, "version": version, "inputs": inputs, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class Manifest:
    """一支影片整個 pipeline 的 provenance 記錄。

    `gpu_model` 與 `library_versions` 刻意不進 stage key 計算——它們對快取
    決策毫無用處，但兩次跑出不同結果時，除錯要靠它們。
    """

    source: str  # Drive file id 或 YouTube video id
    source_md5: str = ""
    stage_keys: dict[str, str] = field(default_factory=dict)
    engine: str = ""
    model_ids: dict[str, str] = field(default_factory=dict)
    model_revisions: dict[str, str] = field(default_factory=dict)
    gpu_model: str = ""
    library_versions: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    degraded_window_count: int = 0

    def is_complete(self, stage: str, key: str) -> bool:
        """這個階段是否已經用「完全相同」的 key 跑過。"""
        return self.stage_keys.get(stage) == key

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        data = json.loads(text)
        return cls(**data)
