import numpy as np
from numpy.typing import NDArray


def A(t: float) -> NDArray[float]:
    """Return the matrix A(t)."""
    return np.array([[1.0 + t, 0.0], [0.0, 1.0 - t]])


def dA_dt(t: float) -> NDArray[float]:
    """Return an derivative of A(t)."""
    return np.array([[-1.0, 0.0], [0.0, 1.0]])


def f(x: NDArray[float], t: float) -> NDArray[float]:
    """Compute f(x; t) = A(t)^{-1} x."""
    return np.linalg.solve(A(t), x)


def df_dt(x: NDArray[float], t: float) -> NDArray[float]:
    """Compute the derivative using a buggy version of d(A^{-1})/dt."""
    return -np.linalg.solve(A(t), dA_dt(t) @ f(x, t))
