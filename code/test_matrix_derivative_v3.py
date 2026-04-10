import numpy as np

from matrix_derivative_v1 import df_dt, f


def finite_difference_df_dt(x, t, h=1.0e-6):
    return (f(x, t + h) - f(x, t - h)) / (2.0 * h)


x = np.array([2.0, 3.0])
t = 0.2

assert np.allclose(df_dt(x, t), np.array([-2.0 / 1.2**2, 3.0 / 0.8**2]))
assert np.allclose(df_dt(x, t), finite_difference_df_dt(x, t), atol=1.0e-8)

