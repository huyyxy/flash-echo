from __future__ import annotations

from pathlib import Path

from flash_echo_pipeline.config import ProjectPaths, persona_dash, render_value
from flash_echo_pipeline.runtime.docker import DockerRuntime
from flash_echo_pipeline.runner import PipelineRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_render_value_resolves_nested_pipeline_variables() -> None:
    context = {
        "persona": "male_white_collar",
        "persona_dash": persona_dash("male_white_collar"),
        "paths": {
            "cleaned_data_dir": "data/filler_prefix/{persona}",
            "deploy_dir": "models/deploy/minimind3-filler-{persona_dash}-v1.0.0",
        },
    }
    resolved_context = render_value(context, context)

    assert resolved_context["paths"]["cleaned_data_dir"] == "data/filler_prefix/male_white_collar"
    assert resolved_context["paths"]["deploy_dir"] == (
        "models/deploy/minimind3-filler-male-white-collar-v1.0.0"
    )


def test_runner_lists_configured_models() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    assert "minimind3" in runner.list_models()
    assert "qwen3_5_0_8b" in runner.list_models()
    assert "finetune" in runner.pipeline_steps("minimind3", "train")
    assert "finetune" in runner.pipeline_steps("qwen3_5_0_8b", "train")


def test_docker_runtime_builds_docker_run_command() -> None:
    runtime = DockerRuntime(
        project_root=PROJECT_ROOT,
        config={
            "image": "ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cpu",
            "platform": "linux/amd64",
            "gpu": "none",
            "workdir": "/workspace",
            "mounts": [{"source": ".", "target": "/workspace"}],
            "env": {"PYTHONPATH": "/workspace/src"},
        },
    )

    command = runtime.command(["python3", "tools/example.py"])

    assert command[:4] == ["docker", "run", "--rm", "--platform"]
    assert "ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cpu" in command
    assert command[-2:] == ["python3", "tools/example.py"]
    assert f"{PROJECT_ROOT}:/workspace" in command


def test_runner_can_build_runtime_image_command() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    command = runner.build_image("docker.cpu", dry_run=True)

    assert command == [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "-t",
        "ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cpu",
        "-f",
        "docker/base/Dockerfile.cpu",
        ".",
    ]


def test_runner_can_push_runtime_image_command() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    command = runner.push_image("docker.cpu", dry_run=True)

    assert command == [
        "docker",
        "push",
        "ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cpu",
    ]


def test_runner_can_pull_runtime_image_command() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    command = runner.pull_image("docker.cpu", dry_run=True)

    assert command == [
        "docker",
        "pull",
        "ccr.ccs.tencentyun.com/huyyxy/flash-echo:base-cpu",
    ]


def test_runner_build_all_orders_base_images_first() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    commands = runner.build_all_images(dry_run=True)
    tags = [command[command.index("-t") + 1].rsplit(":", 1)[-1] for command in commands]

    assert tags == [
        "base-cpu",
        "base-cuda",
        "minimind3-train",
        "minimind3-export",
        "minimind3-infer",
        "qwen3_5_0_8b-train",
        "qwen3_5_0_8b-export",
        "qwen3_5_0_8b-infer",
    ]


def test_runner_pull_all_orders_base_images_first() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    commands = runner.pull_all_images(dry_run=True)
    tags = [command[-1].rsplit(":", 1)[-1] for command in commands]

    assert tags == [
        "base-cpu",
        "base-cuda",
        "minimind3-train",
        "minimind3-export",
        "minimind3-infer",
        "qwen3_5_0_8b-train",
        "qwen3_5_0_8b-export",
        "qwen3_5_0_8b-infer",
    ]


def test_qwen_train_dry_run_uses_filler_prefix_sft_script() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    result = runner.run_step(
        model="qwen3_5_0_8b",
        step_name="finetune",
        persona="male_white_collar",
        runtime_override=None,
        resume=False,
        force=False,
        dry_run=True,
    )

    assert result.status == "dry-run"
    assert "tools/training/train_qwen3_5_sft.py" in result.command
    assert "data/filler_prefix/male_white_collar/train.jsonl" in result.command
