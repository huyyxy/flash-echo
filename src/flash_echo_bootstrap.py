"""Console entry bootstrap for editable installs on broken site paths.

Homebrew Python on macOS often adds ``/usr/local/lib/python3.10/site-packages``
to ``sys.path`` without processing ``.pth`` files there. Editable installs then
record metadata but never expose ``src/`` packages. This module fixes that at
CLI startup by reading the editable project location from distribution metadata.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

_PROJECT_SRC: Optional[Path] = None


def _editable_project_src() -> Optional[Path]:
    global _PROJECT_SRC
    if _PROJECT_SRC is not None:
        return _PROJECT_SRC

    import importlib.metadata as metadata

    try:
        dist = metadata.distribution("flash-echo")
    except metadata.PackageNotFoundError:
        return None

    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return None

    url = json.loads(direct_url).get("url", "")
    if not url.startswith("file:"):
        return None

    parsed = urlparse(url)
    root = Path(unquote(parsed.path))
    src = root / "src"
    if src.is_dir():
        _PROJECT_SRC = src
        return src
    return None


def ensure_src_packages() -> None:
    """Expose ``src/`` packages when editable ``.pth`` hooks were not applied."""
    if importlib.util.find_spec("flash_echo") and importlib.util.find_spec(
        "flash_echo_pipeline"
    ):
        return

    src = _editable_project_src()
    if src is None:
        return

    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def launch_app() -> None:
    ensure_src_packages()
    from flash_echo.app import main

    main()


def launch_pipeline_cli() -> None:
    ensure_src_packages()
    from flash_echo_pipeline.cli import main

    raise SystemExit(main())
