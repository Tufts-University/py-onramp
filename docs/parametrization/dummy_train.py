"""A deliberately tiny script with hard-coded hyperparameters.

This is the "edit the file directly" baseline for the parametrization docs.
"""


def train(width: int, depth: int, lr: float, n_epochs: int, batch_size: int) -> None:
    """Placeholder training function used only for the docs."""
    print("Pretend we trained a model with:")
    print(f"  width={width}")
    print(f"  depth={depth}")
    print(f"  lr={lr}")
    print(f"  n_epochs={n_epochs}")
    print(f"  batch_size={batch_size}")


if __name__ == "__main__":
    WIDTH = 64
    DEPTH = 2
    LR = 3e-3
    N_EPOCHS = 200
    BATCH_SIZE = 256

    train(WIDTH, DEPTH, LR, N_EPOCHS, BATCH_SIZE)
