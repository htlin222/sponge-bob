"""要結構化 JSON 回答的模型呼叫。

`deepseek-*` 走 DeepSeek API（GitHub Actions 上沒有 `claude -p` 的訂閱登入），
其他模型名走本機 `claude -p`。DeepSeek 只有 JSON mode、不強制 schema，所以把 schema
寫進 system prompt；回答的內容由呼叫端（`quotes.accept`、`fix.apply_fixes`）再驗證一次。
"""

from __future__ import annotations

import json
import os

from psr import claude_cli

DEEPSEEK_URL = "https://api.deepseek.com"
ATTEMPTS = 2


class LlmError(RuntimeError):
    """模型沒有回傳 JSON 物件。"""


ERRORS = (claude_cli.ClaudeError, LlmError)  # 呼叫端用來接「模型這次沒答好」


def ask(prompt: str, *, system: str, schema: dict, model: str = "haiku", client=None) -> dict:
    if not model.startswith("deepseek"):
        return claude_cli.ask(prompt, system=system, schema=schema, model=model)
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_URL)
    messages = [
        {"role": "system", "content": f"{system}\n\n只輸出一個 JSON 物件，不要加說明或 markdown，"
                                      f"格式符合這個 JSON Schema：\n{json.dumps(schema, ensure_ascii=False)}"},
        {"role": "user", "content": prompt},
    ]
    last = ""
    for _ in range(ATTEMPTS):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              response_format={"type": "json_object"}, temperature=0)
        last = resp.choices[0].message.content or ""
        try:
            out = json.loads(last)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            return out
    # 不附上回答內容：會出現在公開的 Actions log，而回答裡有字幕原文
    raise LlmError(f"{model} 連續 {ATTEMPTS} 次沒有回傳 JSON 物件（最後一次 {len(last)} 字）")
