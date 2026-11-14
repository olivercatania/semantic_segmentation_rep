"""Train individual-organ models and summarize their saved evaluations."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from dsad_rep.config import IMAGE_SIZE, ORGANS
from dsad_rep.reporting import (
    evaluation_matches,
    evaluation_splits,
    load_evaluation,
    print_summary,
    save_summary,
)


ARCHITECTURES = ("deeplabv3", "segformer")


def run_logged(command, log_path):
    with log_path.open("a", encoding="utf-8") as logfile:
        logfile.write(f"\n=== START: {shlex.join(command)} ===\n")
        logfile.flush()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise SystemExit(130)


def main():
    parser = argparse.ArgumentParser(
        description="Train one model per organ, evaluate each, then summarize results."
    )
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, xpu, or an indexed device such as cuda:1")
    parser.add_argument("--eval-split", choices=("train", "test", "both"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training or writing files.")
    parser.add_argument("--force", action="store_true", help="Retrain all requested models from scratch.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.height, args.width) < 1:
        parser.error("Epochs, batch size, height and width must be positive.")
    if args.num_workers < 0:
        parser.error("Number of workers cannot be negative.")

    jobs = [
        (architecture, organ)
        for architecture in dict.fromkeys(args.architectures)
        for organ in dict.fromkeys(args.organs)
    ]
    splits = evaluation_splits(args.eval_split)
    results = []
    print(f"Scheduled models: {len(jobs)}", flush=True)
    for number, (architecture, organ) in enumerate(jobs, start=1):
        output_dir = args.output_dir / architecture / organ
        marker_path = output_dir / "complete.json"
        best_path = output_dir / "best.pt"
        # Evaluation split and device do not change the trained experiment.
        # Preserve this shape so completion markers from earlier runs still work.
        settings = {
            "architecture": architecture, "organ": organ,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "height": args.height, "width": args.width,
            "num_workers": args.num_workers,
        }
        command = [
            sys.executable, "-u", "-m", "scripts.train_single",
            "--architecture", architecture, "--organ", organ,
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--height", str(args.height), "--width", str(args.width),
            "--num-workers", str(args.num_workers), "--device", args.device,
            "--eval-split", args.eval_split, "--output-dir", str(args.output_dir),
        ]
        print(f"\n[{number}/{len(jobs)}] {architecture} / {organ}", flush=True)
        completed = False
        if not args.force and best_path.is_file():
            try:
                previous = json.loads(marker_path.read_text(encoding="utf-8"))
                completed = isinstance(previous, dict) and all(
                    previous.get(key) == value for key, value in settings.items()
                )
            except (OSError, ValueError):
                pass

        def matches(result):
            return evaluation_matches(
                result, model_type="single", architecture=architecture, organ=organ,
                completed_epochs=args.epochs, image_size=(args.height, args.width), splits=splits,
            )

        result = load_evaluation(output_dir) if completed else None
        if completed and matches(result):
            print("Already trained and evaluated; using saved results.", flush=True)
        else:
            if completed:
                command.append("--evaluate-only")
                print("Training complete; evaluating the saved checkpoint.", flush=True)
            elif args.force:
                command.append("--fresh")
            if args.dry_run:
                print(shlex.join(command), flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            # Keep legacy completion evidence available to --evaluate-only.
            if not completed:
                marker_path.unlink(missing_ok=True)
            log_path = output_dir / "console.log"
            exit_code = run_logged(command, log_path)
            if exit_code != 0:
                raise SystemExit(
                    f"Training/evaluation failed for {architecture}/{organ}; "
                    f"exit code {exit_code}. See {log_path}"
                )
            result = load_evaluation(output_dir)
            if not best_path.is_file() or not matches(result):
                raise RuntimeError(
                    f"The job did not produce a checkpoint and matching evaluation in {output_dir}"
                )
            temporary = marker_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            temporary.replace(marker_path)
            print("Completed training and evaluation.", flush=True)
        results.append({
            **result,
            "evaluations": {split: result["evaluations"][split] for split in splits},
        })

    if not args.dry_run:
        save_summary(args.output_dir, results, name="single_summary")
        print_summary(results, title="All requested single-organ results")


if __name__ == "__main__":
    main()
