"""透過本機 `claude -p` 呼叫模型，走使用者的訂閱額度、不需要 API key。

旗標刻意把 context 壓到最小：不載入 MCP、設定檔、slash command 與工具，
否則每次呼叫都會帶上約 80k tokens 的使用者環境（實測），額度一下就燒光。
`--bare` 能更乾淨，但它只接受 ANTHROPIC_API_KEY，不能用訂閱登入。
"""

from __future__ import annotations

import json
import subprocess


class ClaudeError(RuntimeError):
    """`claude -p` 失敗或沒有回傳結構化輸出。"""


def ask(prompt: str, *, system: str, schema: dict, model: str = "haiku",
        timeout: int = 600) -> dict:
    cmd = [
        "claude", "-p", "--model", model, "--output-format", "json",
        "--tools", "", "--no-session-persistence", "--disable-slash-commands",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "",
        "--system-prompt", system,
        "--json-schema", json.dumps(schema, ensure_ascii=False),
    ]
    try:
        out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ClaudeError(f"claude -p 超過 {timeout} 秒沒有回應") from e
    try:
        result = json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeError(f"claude -p 輸出不是 JSON：{(out.stderr or out.stdout)[-300:]}") from e
    if result.get("is_error") or not isinstance(result.get("structured_output"), dict):
        raise ClaudeError(f"claude -p 失敗：{str(result.get('result'))[-300:]}")
    return result["structured_output"]
