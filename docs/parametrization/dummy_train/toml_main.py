"""Build the dataclass from a TOML config."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from common import ExperimentConfig, train


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the TOML example."""
    parser = argparse.ArgumentParser(
        prog="toml_main.py",
        description="Build an ExperimentConfig from a TOML file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the TOML configuration file.",
    )
    return parser


def load_config(path: Path) -> ExperimentConfig:
    """Load experiment settings from TOML."""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    experiment = data.get("experiment", {})
    return ExperimentConfig(**experiment)


def main() -> int:
    """Load a config file and hand the result to train()."""
    args = build_parser().parse_args()
    config = load_config(args.config)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
