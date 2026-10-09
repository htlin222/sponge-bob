import time
import json
from dataclasses import dataclass, field

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# 設計文件 §10：drive.readonly 讓 token 讀得到使用者能看到的任何檔案（包含
# 既有的 mp4）；drive.file 只涵蓋「本 app 建立或使用者明確選取」的檔案。
# 單靠 drive.file 連來源影片都讀不到（實測：完全看不到既有 mp4）；兩者組合
# 起來，這個 token 在數學上無法修改或刪除任何不是它產生的檔案。
SCOPE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
SCOPE_FILE = "https://www.googleapis.com/auth/drive.file"
SCOPES = [SCOPE_READONLY, SCOPE_FILE]

# 設計文件 §1：影片大小上限 8 GB，preflight 超過直接拒絕。
MAX_VIDEO_SIZE_BYTES = 8 * 1024**3

# 多 GB 影片要串流下載，不能整包進記憶體；chunk 大小取 64MB。
_DOWNLOAD_CHUNK_SIZE = 64 * 1024 * 1024

_PREFLIGHT_FIELDS = "id,name,mimeType,size,md5Checksum,parents"


class DriveError(ValueError):
    """Drive 層操作失敗。訊息會原樣貼回 issue 留言，所以要寫得像給人看的。"""


class PreflightError(DriveError):
    """preflight 檢查未通過（非影片、超過大小上限等）。"""


@dataclass(frozen=True, slots=True)
class FileInfo:
    """`files().get()` 回傳的精簡 metadata。

    md5Checksum 是整條 pipeline 的內容定址指紋（§3），且**不需下載任何一個
    位元組**就拿得到——實測在 5.98 GB 檔案上零傳輸取得。這裡把它當一等公民
    欄位放在最前面附近，避免呼叫端忘記它的存在而去重新下載算 hash。
    """
    id: str
    name: str
    mime_type: str
    size: int
    md5_checksum: str
    parents: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FileMeta:
    """`find_or_create` 的回傳值：寫入（新建或更新）後的檔案身份。"""
    id: str
    name: str
    web_view_link: str | None = None


def load_credentials(token_json: str) -> Credentials:
    """從 JSON 字串載入使用者 OAuth token。

    刻意接受字串而非檔案路徑：CI 環境沒有穩定的本機檔案系統可以放憑證，
    但可以把整份 JSON 存進一個 GitHub Secret、經環境變數傳進來。scopes 固定
    為 drive.readonly + drive.file 這一組（見上方常數的說明）。
    """
    info = json.loads(token_json)
    return Credentials.from_authorized_user_info(info, scopes=SCOPES)


_RETRY_DELAYS = (2, 5, 15, 40)


def with_retry(call, what: str):
    """對 Drive 的單次 API 呼叫做重試。

    存在理由是實測踩到的：一次 CI 執行在 Colab T4 上花了 11 分鐘完成轉錄，
    接著在查詢同名檔案的那個 metadata 請求上撞到 BrokenPipeError，整趟 GPU
    成果全部丟失。連線層的瞬斷是常態而非例外，而重跑的代價在這條 pipeline
    上高得離譜——不重試等於把十幾分鐘的算力押在一個 TCP 連線上。

    退避固定不加 jitter：這條 pipeline 一次只有一個執行者，jitter 只會讓
    行為變得不可重現，換不到任何避免驚群的好處。
    """
    last = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
        try:
            return call()
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError) as exc:
            last = exc
            if delay is None:
                break
            time.sleep(delay)
    raise DriveError(f"{what} 重試 {len(_RETRY_DELAYS)} 次後仍失敗：{last}") from last


def preflight(service, file_id: str) -> FileInfo:
    """跑完整條 pipeline 前的守門檢查：只呼叫一次 `files().get()`，
    不傳輸任何影片資料就驗證 mimeType 與大小是否合法（設計文件 §12）。
    """
    meta = with_retry(
        lambda: service.files().get(fileId=file_id, fields=_PREFLIGHT_FIELDS).execute(),
        f"讀取 {file_id} 的 metadata")

    mime_type = meta.get("mimeType", "")
    if not mime_type.startswith("video/"):
        raise PreflightError(
            f"{meta.get('name', file_id)} 不是影片檔（mimeType={mime_type!r}），拒絕處理。"
        )

    size = int(meta.get("size", 0))
    if size > MAX_VIDEO_SIZE_BYTES:
        raise PreflightError(
            f"{meta.get('name', file_id)} 大小 {size / 1024**3:.2f} GB 超過 8 GB 上限，拒絕處理。"
        )

    return FileInfo(
        id=meta["id"],
        name=meta["name"],
        mime_type=mime_type,
        size=size,
        md5_checksum=meta.get("md5Checksum", ""),
        parents=list(meta.get("parents", [])),
    )


def _escape_query_value(name: str) -> str:
    """Drive `files().list()` 的 `q` 參數是一段小型查詢語言，字串常值用單引號
    包住。反斜線必須先跳脫，否則接著跳脫單引號時產生的反斜線會被誤認成
    使用者原本輸入的一部分而重複跳脫。"""
    return name.replace("\\", "\\\\").replace("'", "\\'")


def find_or_create(service, name: str, folder_id: str, local_path: str, mime_type: str) -> FileMeta:
    """把 `local_path` 的內容寫成 Drive 裡名為 `name`（位於 `folder_id`）的檔案，
    冪等：同名檔案已存在就 update（產生新 revision，舊版仍可還原），
    否則才 create。

    Drive 允許同一資料夾內有同名檔案，`files().create()` 每次都無條件建立新檔
    ——三次重跑會留下三個同名 SRT、三個不同的 webViewLink（設計文件 §11）。
    """
    escaped_name = _escape_query_value(name)
    query = f"name = '{escaped_name}' and '{folder_id}' in parents and trashed = false"
    response = with_retry(
        lambda: service.files()
        .list(q=query, fields="files(id,name,webViewLink)", spaces="drive")
        .execute(),
        f"查詢同名檔案 {name}")
    matches = response.get("files", [])

    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

    if matches:
        file_id = matches[0]["id"]
        result = with_retry(
            lambda: service.files()
            .update(fileId=file_id, media_body=media, fields="id,name,webViewLink")
            .execute(),
            f"更新 {name}")
    else:
        body = {"name": name, "parents": [folder_id]}
        result = with_retry(
            lambda: service.files()
            .create(body=body, media_body=media, fields="id,name,webViewLink")
            .execute(),
            f"建立 {name}"
        )

    return FileMeta(
        id=result["id"],
        name=result.get("name", name),
        web_view_link=result.get("webViewLink"),
    )


def download(service, file_id: str, dest_path: str) -> None:
    """把 `file_id` 的內容串流下載到 `dest_path`。用 chunked
    `MediaIoBaseDownload`，讓多 GB 的影片邊下載邊寫檔，而不是整包灌進記憶體。
    """
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=_DOWNLOAD_CHUNK_SIZE)
        done = False
        while not done:
            _status, done = with_retry(downloader.next_chunk, f"下載 {file_id}")


def mint_readonly_access_token(creds: Credentials) -> str:
    """換發一個只帶 drive.readonly 範圍的裸 access token，交給 Colab VM。

    設計文件 §10：VM 只活一小時、只拿唯讀，**從頭到尾不能看見 refresh
    token**——上傳一律由 runner 執行。做法是另外組一個只含唯讀 scope 的
    Credentials 物件並 refresh 它，回傳的字串本身就不帶 refresh_token，
    交出去之後 VM 沒有任何管道換到更長效或更高權限的憑證。
    """
    readonly_creds = Credentials(
        token=None,
        refresh_token=creds.refresh_token,
        token_uri=creds.token_uri,
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        scopes=[SCOPE_READONLY],
    )
    readonly_creds.refresh(Request())
    return readonly_creds.token


def ensure_folder(service, name: str, parent_id: str | None = None) -> str:
    """取得（必要時建立）一個資料夾，回傳它的 id。

    與 find_or_create 同樣先查再建：Drive 允許同名資料夾共存，直接 create
    會在每次執行時多長出一個同名資料夾，而使用者只會看到一堆長得一樣的東西。
    """
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    clauses = [
        f"name = '{escaped}'",
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
    ]
    if parent_id:
        clauses.append(f"'{parent_id}' in parents")
    found = with_retry(
        lambda: service.files().list(
            q=" and ".join(clauses), fields="files(id)", spaces="drive"
        ).execute(),
        f"查詢資料夾 {name}").get("files", [])
    if found:
        return found[0]["id"]

    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    return with_retry(
        lambda: service.files().create(body=body, fields="id").execute(),
        f"建立資料夾 {name}")["id"]


_LIST_FIELDS = "nextPageToken,files(id,name,mimeType,size,md5Checksum)"


def list_children(service, folder_id: str) -> list[dict]:
    """列出資料夾底下所有未刪除的檔案（含分頁）。

    批次模式一開始只呼叫這個一次就決定整批要做什麼：哪些影片已經有 SRT、
    哪些在 _psr/ 底下留有可重用的轉錄。兩百多集用 pageSize=1000 一頁就拿完，
    比逐集查詢省下數百個 API 呼叫。
    """
    files: list[dict] = []
    token = None
    while True:
        resp = with_retry(
            lambda: service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields=_LIST_FIELDS, pageSize=1000, pageToken=token, spaces="drive",
            ).execute(),
            f"列出資料夾 {folder_id}")
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return files


def read_text(service, file_id: str) -> str:
    """讀回一個小型文字產物（words.json、manifest.json）。"""
    data = with_retry(lambda: service.files().get_media(fileId=file_id).execute(),
                      f"讀取 {file_id}")
    return data.decode("utf-8")
