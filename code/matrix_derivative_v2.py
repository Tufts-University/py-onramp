import numpy as np


def A(t):
    """Return the matrix A(t)."""
    return np.array([[1.0 + t, 0.0], [0.0, 1.0 - t]])


def f(x, t):
    """Compute f(x; t) = A(t)^{-1} x."""
    x = np.asarray(x, dtype=float)
    return np.linalg.solve(A(t), x)


def df_dt(x, t):
    """The corrected analytic derivative."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            -x[0] / (1.0 + t) ** 2,
            x[1] / (1.0 - t) ** 2,
        ]
    )

