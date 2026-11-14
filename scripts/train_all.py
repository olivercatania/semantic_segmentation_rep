"""Run individual-organ and shared-encoder experiments, then report all scores."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=("deeplabv3", "segformer"), default=["deeplabv3", "segformer"])
    parser.add_argument("--organs", nargs="+", choices=ORGANS, default=list(ORGANS), help="Organs for single-organ models; combined models always use all organs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2, help="Single-organ training batch size.")
    parser.add_argument("--multi-batch-size", type=int, default=1, help="Combined-model training batch size.")
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-split", choices=("train", "test", "both"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--force", action="store_true", help="Retrain all requested models from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Print suite commands without running or writing files.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.multi_batch_size, args.height, args.width) < 1:
        parser.error("Epochs, batch sizes, height and width must be positive.")
    if args.num_workers < 0:
        parser.error("Number of workers cannot be negative.")

    architectures = list(dict.fromkeys(args.architectures))
    organs = list(dict.fromkeys(args.organs))
    shared = [
        "--architectures", *architectures,
        "--epochs", str(args.epochs),
        "--height", str(args.height), "--width", str(args.width),
        "--num-workers", str(args.num_workers), "--device", args.device,
        "--eval-split", args.eval_split, "--output-dir", str(args.output_dir),
    ]
    commands = [
        [sys.executable, "-u", "-m", "scripts.train_all_single", *shared,
         "--organs", *organs, "--batch-size", str(args.batch_size),
         *(["--force"] if args.force else [])],
        [sys.executable, "-u", "-m", "scripts.train_multi", *shared,
         "--batch-size", str(args.multi_batch_size),
         *(["--fresh"] if args.force else [])],
    ]
    count = len(architectures) * (len(organs) + 1)
    print(f"Scheduled models: {count}", flush=True)
    for command in commands:
        if args.dry_run:
            print(shlex.join(command), flush=True)
        else:
            subprocess.run(command, check=True)
    if args.dry_run:
        return

    splits = evaluation_splits(args.eval_split)
    results = []
    for architecture in architectures:
        jobs = [
            ("single", organ, args.output_dir / architecture / organ)
            for organ in organs
        ]
        jobs.append(("combined", None, args.output_dir / "combined" / architecture))
        for model_type, organ, output_dir in jobs:
            result = load_evaluation(output_dir)
            if not evaluation_matches(
                result, model_type=model_type, architecture=architecture, organ=organ,
                completed_epochs=args.epochs, image_size=(args.height, args.width), splits=splits,
            ):
                raise RuntimeError(f"Missing or incompatible completed evaluation in {output_dir}")
            results.append({
                **result,
                "evaluations": {split: result["evaluations"][split] for split in splits},
            })
    save_summary(args.output_dir, results)
    print_summary(results, title=f"All {count} model results")


if __name__ == "__main__":
    main()
