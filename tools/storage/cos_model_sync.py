"""Upload and download Flash Echo model artifacts with Tencent Cloud COS.

Credentials are resolved from the process environment first, then from the
project root ``.env`` file:

- ``QCLOUD_SECRET_ID``
- ``QCLOUD_SECRET_KEY``
- optional ``QCLOUD_TOKEN`` for temporary credentials
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKET_URL = "https://weights-1305049745.cos.ap-shanghai.myqcloud.com"


@dataclass(frozen=True)
class Credentials:
    secret_id: str
    secret_key: str
    token: str | None = None


@dataclass(frozen=True)
class CosObject:
    key: str
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Flash Echo model artifacts with Tencent COS.")
    parser.add_argument("action", choices=("upload", "download"))
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--bucket-url", default=DEFAULT_BUCKET_URL)
    parser.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def strip_dotenv_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None:
            return value[:index].rstrip()
    return value


def parse_dotenv_value(raw_value: str) -> str:
    value = strip_dotenv_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            print(f"skip invalid .env line {line_no}: missing '='", file=sys.stderr)
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            print(f"skip invalid .env line {line_no}: empty key", file=sys.stderr)
            continue
        values[key] = parse_dotenv_value(raw_value)
    return values


def resolve_config(name: str, dotenv_values: dict[str, str]) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    return dotenv_values.get(name)


def resolve_credentials(dotenv_path: Path) -> Credentials:
    dotenv_values = load_dotenv(dotenv_path)
    secret_id = resolve_config("QCLOUD_SECRET_ID", dotenv_values)
    secret_key = resolve_config("QCLOUD_SECRET_KEY", dotenv_values)
    token = resolve_config("QCLOUD_TOKEN", dotenv_values)
    if not secret_id:
        raise RuntimeError("missing QCLOUD_SECRET_ID in environment or project .env")
    if not secret_key:
        raise RuntimeError("missing QCLOUD_SECRET_KEY in environment or project .env")
    return Credentials(secret_id=secret_id, secret_key=secret_key, token=token)


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip().strip("/")
    if not prefix:
        raise ValueError("remote-prefix must not be empty")
    return prefix


def quote_path(path: str) -> str:
    return urllib.parse.quote(path, safe="/-_.~")


def quote_query(value: str) -> str:
    return urllib.parse.quote(value, safe="-_.~")


class CosClient:
    def __init__(self, *, bucket_url: str, credentials: Credentials) -> None:
        parsed = urllib.parse.urlparse(bucket_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid bucket-url: {bucket_url}")
        self.bucket_url = bucket_url.rstrip("/")
        self.host = parsed.netloc
        self.credentials = credentials

    def upload_file(self, *, local_path: Path, key: str) -> None:
        data = local_path.read_bytes()
        request = self._request("PUT", key=key, data=data)
        request.add_header("Content-Length", str(len(data)))
        with urllib.request.urlopen(request) as response:
            if response.status not in {200, 201}:
                raise RuntimeError(f"unexpected COS upload status {response.status}: {key}")

    def download_file(self, *, key: str, local_path: Path) -> None:
        request = self._request("GET", key=key)
        with urllib.request.urlopen(request) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected COS download status {response.status}: {key}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(response.read())

    def list_prefix(self, prefix: str) -> list[CosObject]:
        objects: list[CosObject] = []
        marker = ""
        while True:
            params = {"prefix": prefix, "max-keys": "1000"}
            if marker:
                params["marker"] = marker
            payload = self._request_text("GET", key="", params=params)
            root = ET.fromstring(payload)
            namespace = ""
            if root.tag.startswith("{"):
                namespace = root.tag.split("}", 1)[0] + "}"
            page_objects = []
            for item in root.findall(f"{namespace}Contents"):
                key = item.findtext(f"{namespace}Key")
                size_text = item.findtext(f"{namespace}Size") or "0"
                if key is None:
                    continue
                page_objects.append(CosObject(key=key, size=int(size_text)))
            objects.extend(page_objects)

            is_truncated = (root.findtext(f"{namespace}IsTruncated") or "false").lower() == "true"
            if not is_truncated:
                return objects
            next_marker = root.findtext(f"{namespace}NextMarker")
            marker = next_marker or (page_objects[-1].key if page_objects else marker)
            if not marker:
                raise RuntimeError("COS list response is truncated but no next marker was provided")

    def _request_text(self, method: str, *, key: str, params: dict[str, str]) -> str:
        request = self._request(method, key=key, params=params)
        with urllib.request.urlopen(request) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected COS list status {response.status}")
            return response.read().decode("utf-8")

    def _request(
        self,
        method: str,
        *,
        key: str,
        params: dict[str, str] | None = None,
        data: bytes | None = None,
    ) -> urllib.request.Request:
        params = params or {}
        url = self._url(key, params=params)
        headers = {"host": self.host}
        if self.credentials.token:
            headers["x-cos-security-token"] = self.credentials.token
        authorization = self._authorization(method, key=key, params=params, headers=headers)
        request_headers = {key_: value for key_, value in headers.items()}
        request_headers["Authorization"] = authorization
        return urllib.request.Request(url, data=data, headers=request_headers, method=method)

    def _url(self, key: str, *, params: dict[str, str]) -> str:
        path = quote_path(key)
        url = f"{self.bucket_url}/{path}" if path else f"{self.bucket_url}/"
        if params:
            query = "&".join(f"{quote_query(k)}={quote_query(v)}" for k, v in sorted(params.items()))
            url = f"{url}?{query}"
        return url

    def _authorization(
        self,
        method: str,
        *,
        key: str,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> str:
        now = int(time.time())
        sign_time = f"{now};{now + 7200}"
        key_time = sign_time
        http_string = self._http_string(method, key=key, params=params, headers=headers)
        format_string = "sha1\n{}\n{}\n".format(
            sign_time,
            hashlib.sha1(http_string.encode("utf-8")).hexdigest(),
        )
        sign_key = hmac.new(
            self.credentials.secret_key.encode("utf-8"),
            key_time.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()
        signature = hmac.new(sign_key.encode("utf-8"), format_string.encode("utf-8"), hashlib.sha1).hexdigest()
        header_list = ";".join(sorted(headers))
        param_list = ";".join(sorted(params))
        return (
            "q-sign-algorithm=sha1"
            f"&q-ak={self.credentials.secret_id}"
            f"&q-sign-time={sign_time}"
            f"&q-key-time={key_time}"
            f"&q-header-list={header_list}"
            f"&q-url-param-list={param_list}"
            f"&q-signature={signature}"
        )

    def _http_string(
        self,
        method: str,
        *,
        key: str,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> str:
        uri = "/" + quote_path(key)
        canonical_params = "&".join(f"{quote_query(k)}={quote_query(v)}" for k, v in sorted(params.items()))
        canonical_headers = "&".join(
            f"{quote_query(k)}={quote_query(str(headers[k]))}" for k in sorted(headers)
        )
        return f"{method.lower()}\n{uri}\n{canonical_params}\n{canonical_headers}\n"


def iter_local_files(local_dir: Path) -> Iterable[Path]:
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            yield path


def upload_directory(client: CosClient, *, local_dir: Path, remote_prefix: str, dry_run: bool) -> None:
    if not local_dir.is_dir():
        raise FileNotFoundError(f"local directory does not exist: {local_dir}")
    files = list(iter_local_files(local_dir))
    if not files:
        raise RuntimeError(f"local directory has no files to upload: {local_dir}")
    for path in files:
        relative = path.relative_to(local_dir).as_posix()
        key = f"{remote_prefix}/{relative}"
        if dry_run:
            print(f"upload {path} -> cos://{key}")
            continue
        client.upload_file(local_path=path, key=key)
        print(f"uploaded {path} -> cos://{key}")


def download_directory(client: CosClient, *, local_dir: Path, remote_prefix: str, dry_run: bool) -> None:
    objects = client.list_prefix(remote_prefix.rstrip("/") + "/")
    if not objects:
        raise RuntimeError(f"remote prefix has no objects: {remote_prefix}")
    for item in objects:
        relative = item.key.removeprefix(remote_prefix.rstrip("/") + "/")
        if not relative or relative.endswith("/"):
            continue
        local_path = local_dir / relative
        if dry_run:
            print(f"download cos://{item.key} -> {local_path}")
            continue
        client.download_file(key=item.key, local_path=local_path)
        print(f"downloaded cos://{item.key} -> {local_path}")


def main() -> int:
    args = parse_args()
    remote_prefix = normalize_prefix(args.remote_prefix)
    if args.dry_run and args.action == "download":
        print(f"download cos://{remote_prefix}/... -> {args.local_dir}")
        return 0
    if args.dry_run and args.action == "upload":
        if args.local_dir.is_dir():
            for path in iter_local_files(args.local_dir):
                relative = path.relative_to(args.local_dir).as_posix()
                print(f"upload {path} -> cos://{remote_prefix}/{relative}")
        else:
            print(f"upload {args.local_dir}/... -> cos://{remote_prefix}/...")
        return 0
    credentials = resolve_credentials(args.dotenv)
    client = CosClient(bucket_url=args.bucket_url, credentials=credentials)
    if args.action == "upload":
        upload_directory(client, local_dir=args.local_dir, remote_prefix=remote_prefix, dry_run=args.dry_run)
    else:
        download_directory(client, local_dir=args.local_dir, remote_prefix=remote_prefix, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
