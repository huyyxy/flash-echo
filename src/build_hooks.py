"""Setuptools hooks for shipping a real bootstrap module in editable wheels."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from setuptools.command.editable_wheel import editable_wheel as _EditableWheel
from setuptools.command.install import install as _Install

_SRC_DIR = Path(__file__).resolve().parent
_BOOTSTRAP_SRC = _SRC_DIR / "flash_echo_bootstrap.py"
_BOOTSTRAP_NAME = "flash_echo_bootstrap.py"


def _copy_bootstrap_to(install_lib: str | None) -> None:
    if not install_lib or not _BOOTSTRAP_SRC.is_file():
        return
    target = Path(install_lib) / _BOOTSTRAP_NAME
    shutil.copy2(_BOOTSTRAP_SRC, target)


def _append_bootstrap_to_wheel(dist_dir: str | None) -> None:
    if not dist_dir or not _BOOTSTRAP_SRC.is_file():
        return
    for wheel in Path(dist_dir).glob("*.whl"):
        with zipfile.ZipFile(wheel, "a", compression=zipfile.ZIP_STORED) as archive:
            if _BOOTSTRAP_NAME not in archive.namelist():
                archive.write(_BOOTSTRAP_SRC, _BOOTSTRAP_NAME)


class EditableWheelWithBootstrap(_EditableWheel):
    def run(self) -> None:
        super().run()
        _append_bootstrap_to_wheel(getattr(self, "dist_dir", None))


class InstallWithBootstrap(_Install):
    def run(self) -> None:
        super().run()
        _copy_bootstrap_to(self.install_lib)
