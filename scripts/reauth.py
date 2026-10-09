"""重新取得 Drive OAuth token（本機執行一次）。

為什麼需要：OAuth 同意畫面停在「測試中」時，Google 發的 refresh token
**7 天就失效**，排程批次跑不到一週就會開始 invalid_grant。專案已改為
「實際運作中」，這支腳本只在 token 被撤銷或換帳號時才需要再跑。

流程：在 localhost:8765 開 callback server 並印出一行 `AUTH_URL <網址>`。
用 ego-browser（已登入的 Google 帳號）打開那個網址、按「繼續」，callback
回來就寫好 token.json，見 README「憑證」。

    uv run python -u scripts/reauth.py
    gh secret set GOOGLE_OAUTH_TOKEN < ~/.config/polish-screen-record/token.json
"""

import pathlib

from google_auth_oauthlib.flow import InstalledAppFlow

from psr.drive import SCOPES

CONF = pathlib.Path.home() / ".config/polish-screen-record"


def main():
    flow = InstalledAppFlow.from_client_secrets_file(str(CONF / "client_secret.json"), SCOPES)
    creds = flow.run_local_server(
        port=8765, open_browser=False, access_type="offline", prompt="consent",
        authorization_prompt_message="AUTH_URL {url}")
    tmp = CONF / "token.json.new"
    tmp.write_text(creds.to_json(), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(CONF / "token.json")
    print("TOKEN_SAVED", flush=True)


if __name__ == "__main__":
    main()
