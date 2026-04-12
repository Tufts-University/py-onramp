"""Shared helpers for the parametrization examples.

The point of this module is not to implement training logic.
It exists so the argparse, TOML, and Hydra examples can all show the same
shape: collect parameters, build a dataclass, and pass that dataclass into a
single function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    """A small bundle of hyperparameters passed through the example flows."""

    width: int = 64
    depth: int = 2
    lr: float = 3e-3
    n_epochs: int = 200
    batch_size: int = 256
    seed: int = 0
    output_dir: str = "outputs/default"


def validate_config(config: ExperimentConfig) -> None:
    """Keep the examples honest without adding much machinery."""
    if config.width < 1:
        raise ValueError(f"width must be at least 1, got {config.width=}")
    if config.depth < 1:
        raise ValueError(f"depth must be at least 1, got {config.depth=}")
    if config.lr <= 0.0:
        raise ValueError(f"lr must be positive, got {config.lr=}")
    if config.n_epochs < 1:
        raise ValueError(f"n_epochs must be at least 1, got {config.n_epochs=}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {config.batch_size=}")


def train(config: ExperimentConfig) -> None:
    """Placeholder training entrypoint used by the parametrization examples."""
    validate_config(config)
    print("Resolved experiment config:")
    for name, value in asdict(config).items():
        print(f"  {name}: {value}")
    print()
    print("def train(config: ExperimentConfig) is intentionally a placeholder.")
    print("The example is about parameter flow, not model training.")
