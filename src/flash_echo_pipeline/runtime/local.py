from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalRuntime:
    project_root: Path
    config: dict

    def command(self, step_command: list[str]) -> list[str]:
        return step_command

    def run(self, step_command: list[str]) -> int:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.config.get("env", {}).items()})
        workdir = self.project_root / self.config.get("workdir", ".")
        return subprocess.run(step_command, cwd=workdir, env=env, check=False).returncode
