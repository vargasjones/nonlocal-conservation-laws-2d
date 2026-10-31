import numpy as np

from velocity_field import vstat_x_face, vstat_y_face
from parameters import eps, delta_x, delta_y
from lax_friedrichs import Lf, alpha, beta


lambda_x_max = (1/3) * min(
    1 / alpha,
    1 / (2 * eps * Lf + delta_x * np.max(np.abs(vstat_x_face)))
)

lambda_y_max = (1/3) * min(
    1 / beta,
    1 / (2 * eps * Lf + delta_y * np.max(np.abs(vstat_y_face)))
)


delta_tx = lambda_x_max * delta_x
delta_ty = lambda_y_max * delta_y

delta_t = min(delta_tx, delta_ty)


# Actual CFL ratios corresponding to the SAME delta_t
lambda_x = delta_t / delta_x
lambda_y = delta_t / delta_y
# Since lambda_x = delta_t/delta_x and lambda_y = delta_t/delta_y,
# the x- and y-CFL conditions give two upper bounds for the same delta_t:
# delta_t <= delta_tx and delta_t <= delta_ty.
# Hence delta_t must satisfy delta_t <= min(delta_tx, delta_ty).
print(delta_t)