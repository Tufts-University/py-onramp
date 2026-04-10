import numpy as np

from matrix_derivative_v1 import df_dt, f


def finite_difference_df_dt(x, t, h=1.0e-6):
    return (f(x, t + h) - f(x, t - h)) / (2.0 * h)


def test_f_at_zero():
    x = np.array([2.0, 3.0])
    assert np.allclose(f(x, 0.0), np.array([2.0, 3.0]))


def test_df_dt_matches_formula_at_zero():
    x = np.array([2.0, 3.0])
    assert np.allclose(df_dt(x, 0.0), np.array([-2.0, 3.0]))


def test_df_dt_matches_finite_difference():
    x = np.array([2.0, 3.0])
    t = 0.2
    assert np.allclose(df_dt(x, t), finite_difference_df_dt(x, t), atol=1.0e-8)

