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
    assert runner.pipeline_steps("minimind3", "download") == ["download_pretrained"]
    assert runner.pipeline_steps("qwen3_5_0_8b", "download") == ["download_pretrained"]
    assert runner.pipeline_steps("minimind3", "train") == ["download_pretrained", "finetune"]
    assert runner.pipeline_steps("qwen3_5_0_8b", "train") == ["download_pretrained", "finetune"]
    assert runner.pipeline_steps("minimind3", "export") == ["export_onnx"]
    assert runner.pipeline_steps("qwen3_5_0_8b", "export") == ["export_onnx"]
    assert runner.pipeline_steps("minimind3", "infer") == ["serve"]
    assert runner.pipeline_steps("qwen3_5_0_8b", "infer") == ["serve"]
    assert runner.pipeline_steps("minimind3", "upload_checkpoint") == ["upload_checkpoint"]
    assert runner.pipeline_steps("minimind3", "upload_model") == ["upload_deploy"]
    assert runner.pipeline_steps("qwen3_5_0_8b", "download_deploy") == ["download_deploy"]
    assert runner.pipeline_steps("qwen3_5_0_8b", "download_model") == ["download_deploy"]


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


def test_minimind3_infer_uses_infer_runtime_with_port_mapping() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    result = runner.run_step(
        model="minimind3",
        step_name="serve",
        persona="male_white_collar",
        runtime_override=None,
        resume=False,
        force=False,
        dry_run=True,
    )

    assert result.runtime == "docker.minimind3-infer"
    assert "ccr.ccs.tencentyun.com/huyyxy/flash-echo:minimind3-infer" in result.command
    assert "-p" in result.command
    assert "8000:8000" in result.command


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


def test_minimind3_upload_checkpoint_dry_run_uses_cos_prefix() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    result = runner.run_step(
        model="minimind3",
        step_name="upload_checkpoint",
        persona="male_white_collar",
        runtime_override=None,
        resume=False,
        force=False,
        dry_run=True,
    )

    assert result.status == "dry-run"
    assert "tools/storage/cos_model_sync.py" in result.command
    assert "upload" in result.command
    assert "models/checkpoints/minimind3-filler-male_white_collar-v1.0.0/best" in result.command
    assert "flash-echo/minimind3/v1.0.0/male_white_collar/checkpoint/best" in result.command
    assert "https://weights-1305049745.cos.ap-shanghai.myqcloud.com" in result.command


def test_qwen_download_deploy_dry_run_uses_cos_prefix() -> None:
    runner = PipelineRunner(ProjectPaths.discover(PROJECT_ROOT))

    result = runner.run_step(
        model="qwen3_5_0_8b",
        step_name="download_deploy",
        persona="company",
        runtime_override=None,
        resume=False,
        force=False,
        dry_run=True,
    )

    assert result.status == "dry-run"
    assert "tools/storage/cos_model_sync.py" in result.command
    assert "download" in result.command
    assert "models/deploy/qwen3_5_0_8b-filler-company-v1.0.0-onnx" in result.command
    assert "flash-echo/qwen3_5_0_8b/v1.0.0/company/deploy" in result.command
