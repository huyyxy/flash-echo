from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DockerRuntime:
    project_root: Path
    config: dict

    def command(self, step_command: list[str]) -> list[str]:
        docker_command = ["docker", "run", "--rm"]

        platform = self.config.get("platform")
        if platform:
            docker_command.extend(["--platform", str(platform)])

        gpu = self.config.get("gpu")
        if gpu and gpu != "none":
            docker_command.extend(["--gpus", str(gpu)])

        for item in self.config.get("ports", []):
            docker_command.extend(["-p", f"{item['host']}:{item['container']}"])

        for key, value in self.config.get("env", {}).items():
            docker_command.extend(["-e", f"{key}={value}"])

        for item in self.config.get("mounts", []):
            source = Path(item["source"])
            if not source.is_absolute():
                source = self.project_root / source
            docker_command.extend(["-v", f"{source.resolve()}:{item['target']}"])

        workdir = self.config.get("workdir")
        if workdir:
            docker_command.extend(["-w", str(workdir)])

        docker_command.append(str(self.config["image"]))
        docker_command.extend(step_command)
        return docker_command

    def run(self, step_command: list[str]) -> int:
        return subprocess.run(self.command(step_command), cwd=self.project_root, check=False).returncode
