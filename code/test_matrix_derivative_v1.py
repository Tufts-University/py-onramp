import numpy as np

from matrix_derivative_v1 import df_dt, f

x = np.array([2.0, 3.0])
t = 0.2

print("f(x; t) =", f(x, t))
print("df_dt(x, t) =", df_dt(x, t))

