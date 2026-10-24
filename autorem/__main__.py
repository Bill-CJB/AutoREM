"""Command-line entry points for AutoREM training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoREM: memory-augmented robust optimization reformulation.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Adapt memory with the final AutoREM training procedure.")
    train_parser.add_argument("--validation", type=Path, default=Path("datasets/validation.json"))
    train_parser.add_argument("--output", type=Path, default=Path("outputs/training"))
    train_parser.add_argument("--resume", type=Path, help="Resume an epoch checkpoint.")
    evaluate_parser = commands.add_parser("evaluate", help="Evaluate the frozen AutoREM memory.")
    evaluate_parser.add_argument("--dataset", choices=("random", "hard", "large", "application", "general_p", "coupled_row", "external"), default="random")
    evaluate_parser.add_argument("--data", type=Path, help="Use a custom JSON dataset with the selected dataset pipeline.")
    evaluate_parser.add_argument("--memory", type=Path, default=Path("memory/autorem.json"))
    evaluate_parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    evaluate_parser.add_argument("--workers", type=positive_int, default=4)
    evaluate_parser.add_argument("--limit", type=positive_int, help="Evaluate the first N samples for a smoke run.")
    args = parser.parse_args()
    if args.command == "train":
        from .train import train
        train(args)
    else:
        from .evaluate import evaluate
        evaluate(args)


if __name__ == "__main__":
    main()
