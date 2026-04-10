import numpy as np

from matrix_derivative_v1 import df_dt

x = np.array([2.0, 3.0])
t = 0.0

assert np.allclose(df_dt(x, t), np.array([-2.0, 3.0]))

