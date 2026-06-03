from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash_echo_pipeline.config import ProjectPaths, load_yaml, persona_dash, render_value
from flash_echo_pipeline.runtime import DockerRuntime, LocalRuntime


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    command: list[str]
    runtime: str


class PipelineRunner:
    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self.paths = paths or ProjectPaths.discover()

    def list_models(self) -> list[str]:
        return sorted(path.stem for path in self.paths.pipeline_dir.glob("*.yaml"))

    def load_pipeline(self, model: str) -> dict[str, Any]:
        return load_yaml(self.paths.pipeline_dir / f"{model}.yaml")

    def load_runtime(self, runtime_name: str) -> dict[str, Any]:
        return load_yaml(self.paths.runtime_dir / f"{runtime_name}.yaml")

    def pipeline_steps(self, model: str, pipeline_name: str) -> list[str]:
        config = self.load_pipeline(model)
        pipelines = config.get("pipelines", {})
        pipeline_name = self._normalize_name(pipeline_name)
        if pipeline_name not in pipelines:
            known = ", ".join(sorted(pipelines))
            raise KeyError(f"unknown pipeline {model}.{pipeline_name}; known pipelines: {known}")
        return list(pipelines[pipeline_name])

    def run_pipeline(
        self,
        *,
        model: str,
        pipeline_name: str,
        persona: str | None,
        runtime_override: str | None,
        resume: bool,
        force_steps: set[str],
        from_step: str | None,
        dry_run: bool,
    ) -> list[StepResult]:
        steps = self.pipeline_steps(model, pipeline_name)
        if from_step is not None:
            from_step = self._normalize_name(from_step)
            if from_step not in steps:
                raise KeyError(f"--from step is not part of {model}.{pipeline_name}: {from_step}")
            steps = steps[steps.index(from_step) :]
        force_steps = {self._normalize_name(step) for step in force_steps}

        results: list[StepResult] = []
        for step_name in steps:
            result = self.run_step(
                model=model,
                step_name=step_name,
                persona=persona,
                runtime_override=runtime_override,
                resume=resume,
                force=step_name in force_steps,
                dry_run=dry_run,
            )
            results.append(result)
            if result.status == "failed":
                break
        return results

    def run_step(
        self,
        *,
        model: str,
        step_name: str,
        persona: str | None,
        runtime_override: str | None,
        resume: bool,
        force: bool,
        dry_run: bool,
    ) -> StepResult:
        config = self.load_pipeline(model)
        steps = config.get("steps", {})
        step_name = self._normalize_name(step_name)
        if step_name not in steps:
            known = ", ".join(sorted(steps))
            raise KeyError(f"unknown step {step_name} for model {model}; known steps: {known}")

        context = self._context(config, persona)
        rendered_step = render_value(steps[step_name], context)
        runtime_name = runtime_override or rendered_step.get("runtime", "local")
        runtime_config = self.load_runtime(runtime_name)
        runtime = self._runtime(runtime_config)
        step_command = [str(item) for item in rendered_step["command"]]
        full_command = runtime.command(step_command)

        outputs = [Path(item) for item in rendered_step.get("outputs", [])]
        can_resume = bool(rendered_step.get("resume", False)) and resume and outputs
        if can_resume and not force and self._all_exist(outputs):
            self._print_step(step_name, runtime_name, full_command, status="skipped")
            return StepResult(step_name, "skipped", full_command, runtime_name)

        if dry_run:
            self._print_step(step_name, runtime_name, full_command, status="dry-run")
            return StepResult(step_name, "dry-run", full_command, runtime_name)

        missing_inputs = self._missing_paths(rendered_step.get("inputs", []))
        if missing_inputs:
            message = ", ".join(str(path) for path in missing_inputs)
            raise FileNotFoundError(f"step {step_name} missing inputs: {message}")

        self._print_step(step_name, runtime_name, full_command, status="running")
        return_code = runtime.run(step_command)
        if return_code != 0:
            return StepResult(step_name, "failed", full_command, runtime_name)

        missing_outputs = self._missing_paths(outputs)
        if missing_outputs:
            message = ", ".join(str(path) for path in missing_outputs)
            raise FileNotFoundError(f"step {step_name} finished but outputs are missing: {message}")

        return StepResult(step_name, "completed", full_command, runtime_name)

    def build_image(self, image_ref: str, *, dry_run: bool) -> list[str]:
        runtime_path = self.paths.runtime_dir / f"{image_ref}.yaml"
        if runtime_path.is_file():
            runtime_config = load_yaml(runtime_path)
            command = self._docker_build_command(runtime_config)
            if dry_run:
                print(self._format_command(command))
                return command
            return_code = subprocess.run(command, cwd=self.paths.root, check=False).returncode
            if return_code != 0:
                raise RuntimeError(f"image build failed for runtime {image_ref}")
            return command

        model, image_name = self._split_ref(image_ref)
        config = self.load_pipeline(model)
        for step_name, step in config.get("steps", {}).items():
            if step.get("image") != image_ref and step.get("image") != image_name:
                continue
            command = [str(item) for item in render_value(step["command"], self._context(config, None))]
            if dry_run:
                print(self._format_command(command))
                return command
            return_code = subprocess.run(command, cwd=self.paths.root, check=False).returncode
            if return_code != 0:
                raise RuntimeError(f"image build failed for {image_ref}")
            return command
        raise KeyError(f"unknown image build target: {image_ref}")

    def build_all_images(self, *, dry_run: bool) -> list[list[str]]:
        commands = []
        for image_ref in self.list_image_refs():
            commands.append(self.build_image(image_ref, dry_run=dry_run))
        return commands

    def pull_image(self, image_ref: str, *, dry_run: bool) -> list[str]:
        image = self._resolve_image(image_ref)
        command = ["docker", "pull", image]
        if dry_run:
            print(self._format_command(command))
            return command

        return_code = subprocess.run(command, cwd=self.paths.root, check=False).returncode
        if return_code != 0:
            raise RuntimeError(f"image pull failed for {image_ref}")
        return command

    def pull_all_images(self, *, dry_run: bool) -> list[list[str]]:
        commands = []
        for image_ref in self.list_image_refs():
            commands.append(self.pull_image(image_ref, dry_run=dry_run))
        return commands

    def push_image(self, image_ref: str, *, dry_run: bool) -> list[str]:
        image = self._resolve_image(image_ref)
        command = ["docker", "push", image]
        if dry_run:
            print(self._format_command(command))
            return command

        return_code = subprocess.run(command, cwd=self.paths.root, check=False).returncode
        if return_code != 0:
            raise RuntimeError(f"image push failed for {image_ref}")
        return command

    def push_all_images(self, *, dry_run: bool) -> list[list[str]]:
        commands = []
        for image_ref in self.list_image_refs():
            commands.append(self.push_image(image_ref, dry_run=dry_run))
        return commands

    def list_image_refs(self) -> list[str]:
        refs = [path.stem for path in sorted(self.paths.runtime_dir.glob("docker.*.yaml"))]
        for model in self.list_models():
            config = self.load_pipeline(model)
            for step in config.get("steps", {}).values():
                image_ref = step.get("image")
                if image_ref:
                    refs.append(str(image_ref))
        refs = sorted(dict.fromkeys(refs), key=self._image_ref_sort_key)

        unique_refs = []
        seen_images = set()
        for ref in refs:
            image = self._resolve_image(ref)
            if image in seen_images:
                continue
            seen_images.add(image)
            unique_refs.append(ref)
        return unique_refs

    def _docker_build_command(self, runtime_config: dict[str, Any]) -> list[str]:
        dockerfile = runtime_config.get("dockerfile")
        image = runtime_config.get("image")
        if not dockerfile or not image:
            raise ValueError("runtime image build requires both dockerfile and image")

        command = ["docker", "build"]
        platform = runtime_config.get("platform")
        if platform:
            command.extend(["--platform", str(platform)])
        command.extend(["-t", str(image), "-f", str(dockerfile), "."])
        return command

    def _resolve_image(self, image_ref: str) -> str:
        runtime_path = self.paths.runtime_dir / f"{image_ref}.yaml"
        if runtime_path.is_file():
            image = load_yaml(runtime_path).get("image")
            if image:
                return str(image)

        model, image_name = self._split_ref(image_ref)
        config = self.load_pipeline(model)
        for step in config.get("steps", {}).values():
            if step.get("image") != image_ref and step.get("image") != image_name:
                continue
            command = [str(item) for item in render_value(step["command"], self._context(config, None))]
            for index, item in enumerate(command):
                if item in {"-t", "--tag"} and index + 1 < len(command):
                    return command[index + 1]
        raise KeyError(f"unknown image target: {image_ref}")

    def _context(self, config: dict[str, Any], persona: str | None) -> dict[str, Any]:
        resolved_persona = persona or config.get("default_persona", "")
        context = dict(config)
        context["persona"] = resolved_persona
        context["persona_dash"] = persona_dash(resolved_persona)
        context = render_value(context, context)
        return context

    def _runtime(self, runtime_config: dict[str, Any]) -> LocalRuntime | DockerRuntime:
        runtime_type = runtime_config.get("type")
        if runtime_type == "local":
            return LocalRuntime(project_root=self.paths.root, config=runtime_config)
        if runtime_type == "docker":
            return DockerRuntime(project_root=self.paths.root, config=runtime_config)
        raise ValueError(f"unknown runtime type: {runtime_type}")

    def _missing_paths(self, paths: list[str | Path]) -> list[Path]:
        missing: list[Path] = []
        for path in paths:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = self.paths.root / resolved
            if not resolved.exists():
                missing.append(resolved)
        return missing

    def _all_exist(self, paths: list[Path]) -> bool:
        return not self._missing_paths(paths)

    def _print_step(self, step_name: str, runtime_name: str, command: list[str], *, status: str) -> None:
        print(f"[{status}] {step_name} ({runtime_name})")
        print(self._format_command(command))

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return " ".join(shlex.quote(item) for item in command)

    @staticmethod
    def _split_ref(ref: str) -> tuple[str, str]:
        if "." not in ref:
            raise ValueError("expected ref in the form <model>.<name>")
        model, name = ref.split(".", 1)
        return model, name

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("-", "_")

    def _image_ref_sort_key(self, image_ref: str) -> tuple[int, str]:
        try:
            image = self._resolve_image(image_ref)
        except Exception:  # noqa: BLE001 - unknown refs should sort last, then fail later.
            return (999, image_ref)
        tag = image.rsplit(":", 1)[-1]
        order = {
            "base-cpu": 0,
            "base-cuda": 1,
            "minimind3-train": 2,
            "minimind3-export": 3,
            "minimind3-infer": 4,
            "qwen3_5_0_8b-train": 5,
            "qwen3_5_0_8b-export": 6,
            "qwen3_5_0_8b-infer": 7,
        }
        return (order.get(tag, 500), image_ref)
