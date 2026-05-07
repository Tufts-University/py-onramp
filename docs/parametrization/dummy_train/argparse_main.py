"""Pass hyperparameters from argparse into a dataclass."""

from __future__ import annotations

import argparse

from common import ExperimentConfig, train


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the simple argparse example."""
    parser = argparse.ArgumentParser(
        prog="argparse_main.py",
        description="Build an ExperimentConfig from command-line arguments.",
    )
    parser.add_argument(
        "--width", type=int, default=64, help="Hidden layer width in the dummy model."
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="Number of hidden layers."
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="outputs/argparse",
        help="Directory name recorded in the config.",
    )
    return parser


def main() -> int:
    """Parse CLI arguments, build the config, and hand it to train()."""
    args = build_parser().parse_args()
    config = ExperimentConfig(
        width=args.width,
        depth=args.depth,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
