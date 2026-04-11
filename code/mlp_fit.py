"""
Fit a small MLP to f(x) = x1*(1-x1)*cos(4*pi*x1)*sin^2(4*pi*x2^2)
using JAX, Equinox, and Optax.

Intentionally has no CLI parsing, TOML config, or Hydra.
Those are layered on in subsequent examples that import from this file.
"""

import dataclasses
import json
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Key = PRNGKeyArray
MLP = eqx.nn.MLP
Inputs = Float[Array, "n 2"]
Targets = Float[Array, "n"]
Loss = Float[Array, ""]
BatchInputs = Float[Array, "b 2"]
BatchTargets = Float[Array, "b"]
TrainInputs = Float[Array, "n_train 2"]
TrainTargets = Float[Array, "n_train"]
ValInputs = Float[Array, "n_val 2"]
ValTargets = Float[Array, "n_val"]
TestInputs = Float[Array, "n_test 2"]
TestTargets = Float[Array, "n_test"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OptStats:
    train_losses: list[float]
    val_losses: list[float]
    final_train_loss: float
    final_val_loss: float
    final_test_loss: float | None
    n_epochs: int
    lr: float
    batch_size: int
    n_train: int
    n_val: int
    n_test: int


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def generate_data(
    n_points: int, key: Key, noise_std: float = 0.0
) -> tuple[Inputs, Targets]:
    """
    Sample n_points uniformly from [0, 1]^2 and evaluate the target function.

    Returns (X, y) where X has shape (n_points, 2) and y has shape (n_points,).
    """
    x = jax.random.uniform(key, shape=(n_points, 2))
    x1, x2 = x[:, 0], x[:, 1]
    y = (
        x1
        * (1.0 - x1)
        * jnp.cos(4.0 * jnp.pi * x1)
        * jnp.sin(4.0 * jnp.pi * x2**2) ** 2
    )
    if noise_std > 0.0:
        noise_key, _ = jax.random.split(key)
        y = y + jax.random.normal(noise_key, shape=y.shape) * noise_std
    return x, y


def make_splits(
    x: Inputs, y: Targets, test_size: float, val_size: float, key: Key
) -> tuple[TrainInputs, TrainTargets, ValInputs, ValTargets, TestInputs, TestTargets]:
    """Split arrays into deterministic train/validation/test sets."""
    if (val_size + test_size) > 1.0:
        raise ValueError(
            f"Must have val_size + test_size <= 1.0, got {val_size+test_size=}"
        )
    n_points = len(x)
    n_test = int(round(n_points * test_size))
    n_val = int(round(n_points * val_size))
    n_train = n_points - n_val - n_test
    if n_train == 0:
        raise ValueError(f"Selected test_size and val_size results in {n_train=}!")

    indices = jax.random.permutation(key, n_points)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]


@eqx.filter_jit
def evaluate(mlp: MLP, x: Inputs, y: Targets) -> Loss:
    """Compute full-dataset mean squared error."""
    pred = jax.vmap(mlp)(x)
    return jnp.mean((pred - y) ** 2)


# ---------------------------------------------------------------------------
# Training Hot Path
# ---------------------------------------------------------------------------


def train(
    mlp: MLP,
    x_train: Inputs,
    y_train: Targets,
    x_val: Inputs,
    y_val: Targets,
    lr: float,
    n_epochs: int,
    batch_size: int,
    key: Key,
) -> tuple[MLP, OptStats]:
    """
    Train mlp with Adam to minimise MSE on (x_train, y_train), evaluating on
    (x_val, y_val) each epoch.

    Each epoch shuffles the training set with a JAX permutation and then walks
    through contiguous mini-batches, keeping the final short batch if needed.

    Returns the trained model and an OptStats with per-epoch loss history.
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(mlp, eqx.is_array))

    @eqx.filter_jit
    def step(
        mlp: MLP,
        opt_state: optax.OptState,
        xb: BatchInputs,
        yb: BatchTargets,
    ) -> tuple[MLP, optax.OptState, Loss]:
        loss, grads = eqx.filter_value_and_grad(evaluate)(mlp, xb, yb)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(mlp, eqx.is_array)
        )
        return eqx.apply_updates(mlp, updates), new_opt_state, loss

    n_train = len(x_train)
    epoch_keys = jax.random.split(key, n_epochs)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch_key in epoch_keys:
        permutation = jax.random.permutation(epoch_key, n_train)
        x_epoch = x_train[permutation]
        y_epoch = y_train[permutation]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            stop = min(start + batch_size, n_train)
            xb = x_epoch[start:stop]
            yb = y_epoch[start:stop]
            mlp, opt_state, loss = step(mlp, opt_state, xb, yb)
            epoch_loss += float(loss)
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))
        val_losses.append(float(evaluate(mlp, x_val, y_val)))

    stats = OptStats(
        train_losses=train_losses,
        val_losses=val_losses,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        final_test_loss=None,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        n_train=len(x_train),
        n_val=len(x_val),
        n_test=0,
    )
    return mlp, stats


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hyperparameters — no CLI, no config file (yet)
    N_TOTAL = 5000
    VAL_SIZE = 0.2
    TEST_SIZE = 0.2
    HIDDEN = [64, 64]
    LR = 3e-3
    N_EPOCHS = 200
    BATCH_SIZE = 256
    SEED = 0

    key = jax.random.PRNGKey(SEED)
    data_key, split_key, model_key, train_key = jax.random.split(key, 4)

    x, y = generate_data(N_TOTAL, data_key)

    x_train, y_train, x_val, y_val, x_test, y_test = make_splits(
        x,
        y,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        key=split_key,
    )

    mlp = cast(
        MLP,
        eqx.nn.MLP(
            in_size=2,
            out_size="scalar",
            width_size=HIDDEN[0],
            depth=len(HIDDEN),
            activation=jax.nn.tanh,
            key=key,
        ),
    )
    mlp, stats = train(
        mlp, x_train, y_train, x_val, y_val, LR, N_EPOCHS, BATCH_SIZE, train_key
    )
    test_loss = float(evaluate(mlp, x_test, y_test))
    stats.n_test = len(x_test)
    stats.final_test_loss = test_loss

    # Save model weights (equinox native format)
    eqx.tree_serialise_leaves("mlp.eqx", mlp)
    print("Model saved to mlp.eqx")

    # Save stats as JSON
    with open("opt_stats.json", "w") as f:
        json.dump(dataclasses.asdict(stats), f, indent=2)
    print("Stats saved to opt_stats.json")
    print(f"  final train loss : {stats.final_train_loss:.6f}")
    print(f"  final val   loss : {stats.final_val_loss:.6f}")
    print(f"  final test  loss : {stats.final_test_loss:.6f}")
