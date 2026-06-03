from __future__ import annotations

import argparse
from typing import Sequence

from flash_echo_pipeline.runner import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flash-echo-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List configured models and pipelines.")
    list_parser.add_argument("--model", default=None)

    run_parser = subparsers.add_parser("run", help="Run a configured pipeline.")
    run_parser.add_argument("pipeline", help="Pipeline ref, for example minimind3.full.")
    run_parser.add_argument("--persona", default=None)
    run_parser.add_argument("--runtime", default=None, help="Override every step runtime.")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--force", action="append", default=[], help="Force one step to rerun.")
    run_parser.add_argument("--from", dest="from_step", default=None, help="Start from the named step.")
    run_parser.add_argument("--dry-run", action="store_true")

    step_parser = subparsers.add_parser("step", help="Run a single step.")
    step_parser.add_argument("step")
    step_parser.add_argument("--model", required=True)
    step_parser.add_argument("--persona", default=None)
    step_parser.add_argument("--runtime", default=None)
    step_parser.add_argument("--resume", action="store_true")
    step_parser.add_argument("--force", action="store_true")
    step_parser.add_argument("--dry-run", action="store_true")

    image_parser = subparsers.add_parser("image", help="Manage Docker images.")
    image_subparsers = image_parser.add_subparsers(dest="image_command", required=True)
    build_parser_ = image_subparsers.add_parser("build", help="Build a configured image.")
    build_parser_.add_argument("image")
    build_parser_.add_argument("--dry-run", action="store_true")
    build_all_parser_ = image_subparsers.add_parser("build-all", help="Build every configured image.")
    build_all_parser_.add_argument("--dry-run", action="store_true")
    push_parser_ = image_subparsers.add_parser("push", help="Push a configured image.")
    push_parser_.add_argument("image")
    push_parser_.add_argument("--dry-run", action="store_true")
    push_all_parser_ = image_subparsers.add_parser("push-all", help="Push every configured image.")
    push_all_parser_.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner = PipelineRunner()

    try:
        if args.command == "list":
            _list(runner, args.model)
            return 0

        if args.command == "run":
            model, pipeline_name = _split_ref(args.pipeline)
            results = runner.run_pipeline(
                model=model,
                pipeline_name=pipeline_name,
                persona=args.persona,
                runtime_override=args.runtime,
                resume=args.resume,
                force_steps=set(args.force),
                from_step=args.from_step,
                dry_run=args.dry_run,
            )
            return 1 if any(result.status == "failed" for result in results) else 0

        if args.command == "step":
            result = runner.run_step(
                model=args.model,
                step_name=args.step,
                persona=args.persona,
                runtime_override=args.runtime,
                resume=args.resume,
                force=args.force,
                dry_run=args.dry_run,
            )
            return 1 if result.status == "failed" else 0

        if args.command == "image" and args.image_command == "build":
            runner.build_image(args.image, dry_run=args.dry_run)
            return 0

        if args.command == "image" and args.image_command == "build-all":
            runner.build_all_images(dry_run=args.dry_run)
            return 0

        if args.command == "image" and args.image_command == "push":
            runner.push_image(args.image, dry_run=args.dry_run)
            return 0

        if args.command == "image" and args.image_command == "push-all":
            runner.push_all_images(dry_run=args.dry_run)
            return 0
    except Exception as exc:  # noqa: BLE001 - CLI should print friendly errors.
        parser.exit(1, f"flash-echo-pipeline: {exc}\n")

    parser.error("unknown command")
    return 2


def _list(runner: PipelineRunner, model: str | None) -> None:
    models = [model] if model else runner.list_models()
    for item in models:
        config = runner.load_pipeline(item)
        pipelines = ", ".join(sorted(config.get("pipelines", {})))
        steps = ", ".join(sorted(config.get("steps", {})))
        print(f"{item}")
        print(f"  pipelines: {pipelines}")
        print(f"  steps: {steps}")


def _split_ref(ref: str) -> tuple[str, str]:
    if "." not in ref:
        raise ValueError("expected ref in the form <model>.<pipeline>")
    return tuple(ref.split(".", 1))  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
