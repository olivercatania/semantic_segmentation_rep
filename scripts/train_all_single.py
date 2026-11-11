
"""Train all 22 DSAD single-organ segmentation models sequentially."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dsad_rep.config import IMAGE_SIZE, ORGANS


ARCHITECTURES = ("deeplabv3", "segformer")


def main():
    parser = argparse.ArgumentParser(
        description="Train one model per organ for DeepLabv3 and SegFormer."
    )

    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )

    parser.add_argument(
        "--organs",
        nargs="+",
        choices=ORGANS,
        default=list(ORGANS),
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs without training.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain all requested models from scratch.",
    )

    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.height,
        args.width,
    ) < 1:
        parser.error(
            "Epochs, batch size, height and width must be positive."
        )

    if args.num_workers < 0:
        parser.error("Number of workers cannot be negative.")

    # Construct the list of model/organ combinations
    jobs = [
        (architecture, organ)
        for architecture in dict.fromkeys(args.architectures)
        for organ in dict.fromkeys(args.organs)
    ]

    print(f"Scheduled models: {len(jobs)}", flush=True)

    # Train one model at a time
    for number, (architecture, organ) in enumerate(jobs, start=1):

        output_dir = Path("checkpoints") / architecture / organ

        marker_path = output_dir / "complete.json"
        best_path = output_dir / "best.pt"

        settings = {
            "architecture": architecture,
            "organ": organ,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "num_workers": args.num_workers,
        }

        # Command for training one model
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.train_single",
            "--architecture", architecture,
            "--organ", organ,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--height", str(args.height),
            "--width", str(args.width),
        ]

        print(
            f"\n[{number}/{len(jobs)}] "
            f"{architecture} / {organ}",
            flush=True,
        )

        # Skip models that already completed with these settings
        if (
            not args.force
            and marker_path.is_file()
            and best_path.is_file()
        ):
            try:
                completed_settings = json.loads(
                    marker_path.read_text()
                )

            except (OSError, json.JSONDecodeError):
                completed_settings = None

            if completed_settings == settings:
                print(
                    "Already completed; skipping.",
                    flush=True,
                )
                continue

        # Preview commands without running them
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Remove the completion marker before starting
        marker_path.unlink(missing_ok=True)

        # Execute the training script and record its output
        log_path = output_dir / "console.log"

        with log_path.open("a", encoding="utf-8") as logfile:

            logfile.write(f"\n=== START: {settings} ===\n")
            logfile.flush()

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            try:
                for line in process.stdout:

                    # Display output in the terminal
                    print(line, end="", flush=True)

                    # Also save output to a log
                    logfile.write(line)
                    logfile.flush()

                exit_code = process.wait()

            except KeyboardInterrupt:

                if process.poll() is None:
                    process.terminate()

                    try:
                        process.wait(timeout=10)

                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                raise SystemExit(130)

        # Stop if a model fails
        if exit_code != 0:
            raise SystemExit(
                f"Training failed for "
                f"{architecture}/{organ}; "
                f"exit code {exit_code}. "
                f"See {log_path}"
            )

        # Verify that training produced a checkpoint
        if not best_path.is_file():
            raise RuntimeError(
                f"Training exited successfully "
                f"but did not produce {best_path}"
            )

        # Mark the model as completed
        temp_marker = marker_path.with_suffix(".tmp")

        temp_marker.write_text(
            json.dumps(settings, indent=2) + "\n"
        )

        temp_marker.replace(marker_path)

        print(
            "Completed successfully.",
            flush=True,
        )

    print(
        "\nAll requested models completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
