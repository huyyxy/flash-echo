"""Download company robot dialogue JSON logs from Aliyun OSS.

日志存储在 ``robot-dialogue.oss-cn-shanghai.aliyuncs.com``，路径结构为::

    {group_id}/{robot_id}/dialogue_*.json

列举 bucket 需要 OSS 访问密钥；单个对象通常可通过 HTTPS 匿名读取。

在项目根目录 ``.env`` 中配置（推荐）::

    OSS_ACCESS_KEY_ID=...
    OSS_ACCESS_KEY_SECRET=...

也支持 ``ALIYUN_ACCESS_KEY_ID`` / ``ALIYUN_ACCESS_KEY_SECRET`` 命名，或命令行 ``--access-key-id`` /
``--access-key-secret``。

配置优先级：命令行参数 > 当前进程环境变量 > 项目根目录 ``.env``。

使用示例::

    pip3 install oss2

    # 密钥写入项目根目录 .env 后直接运行
    python3 tools/training/download_company_dialogue_logs.py

    # 只下载指定组 / 机器人
    python3 tools/training/download_company_dialogue_logs.py \\
      --group-id 0f888b18-6473-4956-954f-a74348c71772 \\
      --robot-id 2dd64e7fba1866c7358e1a8c228fe357

    # 试跑：最多下载 10 个文件
    python3 tools/training/download_company_dialogue_logs.py --max-files 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUCKET = "robot-dialogue"
DEFAULT_ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/raw/company"
PUBLIC_BASE_URL = "https://robot-dialogue.oss-cn-shanghai.aliyuncs.com"


def load_dotenv(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for line_no, raw_line in enumerate(dotenv_path.read_text(encoding="utf-8").splitlines(), start=1):
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
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def resolve_env(name: str, dotenv_values: dict[str, str], default: str | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if name in dotenv_values and dotenv_values[name]:
        return dotenv_values[name]
    return default


def resolve_access_key_id(dotenv_values: dict[str, str], cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    for name in ("OSS_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_ID"):
        value = resolve_env(name, dotenv_values)
        if value:
            return value
    raise SystemExit(
        "missing OSS access key id. Set OSS_ACCESS_KEY_ID (or ALIYUN_ACCESS_KEY_ID) "
        "via environment, project .env, or --access-key-id."
    )


def resolve_access_key_secret(dotenv_values: dict[str, str], cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    for name in ("OSS_ACCESS_KEY_SECRET", "ALIYUN_ACCESS_KEY_SECRET"):
        value = resolve_env(name, dotenv_values)
        if value:
            return value
    raise SystemExit(
        "missing OSS access key secret. Set OSS_ACCESS_KEY_SECRET (or ALIYUN_ACCESS_KEY_SECRET) "
        "via environment, project .env, or --access-key-secret."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download robot dialogue JSON logs from company Aliyun OSS bucket."
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"OSS bucket name (default: {DEFAULT_BUCKET}).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"OSS endpoint URL (default: {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Only download logs under this group id (first path segment).",
    )
    parser.add_argument(
        "--robot-id",
        default=None,
        help="Only download logs under this robot id (second path segment). Requires --group-id.",
    )
    parser.add_argument(
        "--access-key-id",
        default=None,
        help="OSS access key id. Priority: CLI > environment > project .env.",
    )
    parser.add_argument(
        "--access-key-secret",
        default=None,
        help="OSS access key secret. Priority: CLI > environment > project .env.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers (default: 8).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after processing at most N JSON files (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even when the local file already exists.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List matching object keys without downloading.",
    )
    parser.add_argument(
        "--use-sdk-download",
        action="store_true",
        help="Download via oss2 SDK instead of anonymous HTTPS (requires credentials either way for listing).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.robot_id and not args.group_id:
        raise SystemExit("--robot-id requires --group-id.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    if args.max_files is not None and args.max_files < 1:
        raise SystemExit("--max-files must be >= 1.")


def build_list_prefix(group_id: str | None, robot_id: str | None) -> str:
    if group_id and robot_id:
        return f"{group_id}/{robot_id}/"
    if group_id:
        return f"{group_id}/"
    return ""


def is_dialogue_json_key(key: str) -> bool:
    parts = key.split("/")
    if len(parts) != 3:
        return False
    filename = parts[2]
    return filename.endswith(".json") and filename.startswith("dialogue_")


def iter_object_keys(
    *,
    bucket_name: str,
    endpoint: str,
    access_key_id: str,
    access_key_secret: str,
    prefix: str,
    max_files: int | None,
) -> list[tuple[str, int]]:
    try:
        import oss2
    except ImportError as exc:
        raise SystemExit("oss2 is required. Install with: pip3 install oss2") from exc

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    matches: list[tuple[str, int]] = []
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        key = obj.key
        if not is_dialogue_json_key(key):
            continue
        size = int(obj.size or 0)
        matches.append((key, size))
        if max_files is not None and len(matches) >= max_files:
            break
    return matches


def local_path_for_key(output_dir: Path, key: str) -> Path:
    return output_dir / Path(*key.split("/"))


def should_skip(local_path: Path, remote_size: int, overwrite: bool) -> bool:
    if overwrite or not local_path.exists():
        return False
    if remote_size <= 0:
        return local_path.stat().st_size > 0
    return local_path.stat().st_size == remote_size


def download_via_https(key: str, local_path: Path) -> None:
    url = f"{PUBLIC_BASE_URL}/{key}"
    request = Request(url, headers={"User-Agent": "flash-echo/company-dialogue-downloader"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    local_path.write_bytes(data)


def download_via_sdk(
    *,
    bucket,
    key: str,
    local_path: Path,
) -> None:
    bucket.get_object_to_file(key, str(local_path))


def download_one(
    *,
    key: str,
    remote_size: int,
    output_dir: Path,
    overwrite: bool,
    use_sdk_download: bool,
    bucket,
) -> tuple[str, str]:
    local_path = local_path_for_key(output_dir, key)
    if should_skip(local_path, remote_size, overwrite):
        return key, "skipped"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".part")
    try:
        if use_sdk_download:
            download_via_sdk(bucket=bucket, key=key, local_path=temp_path)
        else:
            download_via_https(key, temp_path)
        temp_path.replace(local_path)
    except (HTTPError, URLError, OSError):
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    return key, "downloaded"


def run_downloads(
    objects: Iterable[tuple[str, int]],
    *,
    output_dir: Path,
    overwrite: bool,
    workers: int,
    use_sdk_download: bool,
    bucket,
) -> tuple[int, int, int]:
    downloaded = 0
    skipped = 0
    failed = 0
    total = len(objects) if isinstance(objects, list) else None
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one,
                key=key,
                remote_size=remote_size,
                output_dir=output_dir,
                overwrite=overwrite,
                use_sdk_download=use_sdk_download,
                bucket=bucket,
            ): key
            for key, remote_size in objects
        }
        for index, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            try:
                _, status = future.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                print(f"[fail] {key}: {exc}", file=sys.stderr)
                continue

            if status == "downloaded":
                downloaded += 1
            else:
                skipped += 1

            if total:
                if index % 100 == 0 or index == total:
                    elapsed = time.time() - started
                    print(
                        f"progress {index}/{total} "
                        f"(downloaded={downloaded}, skipped={skipped}, failed={failed}, "
                        f"elapsed={elapsed:.1f}s)"
                    )

    return downloaded, skipped, failed


def main() -> None:
    args = parse_args()
    validate_args(args)

    dotenv_values = load_dotenv(PROJECT_ROOT / ".env")
    access_key_id = resolve_access_key_id(dotenv_values, args.access_key_id)
    access_key_secret = resolve_access_key_secret(dotenv_values, args.access_key_secret)
    prefix = build_list_prefix(args.group_id, args.robot_id)

    print(f"listing oss://{args.bucket}/{prefix or ''} ...")
    objects = iter_object_keys(
        bucket_name=args.bucket,
        endpoint=args.endpoint,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        prefix=prefix,
        max_files=args.max_files,
    )
    print(f"found {len(objects)} dialogue json file(s)")

    if args.list_only:
        for key, size in objects:
            print(f"{key}\t{size}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bucket = None
    if args.use_sdk_download:
        import oss2

        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, args.endpoint, args.bucket)

    downloaded, skipped, failed = run_downloads(
        objects,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        workers=args.workers,
        use_sdk_download=args.use_sdk_download,
        bucket=bucket,
    )
    print(
        f"done: downloaded={downloaded}, skipped={skipped}, failed={failed}, "
        f"output_dir={args.output_dir}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
