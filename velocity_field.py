import numpy as np
from parameters import Nx, Ny


# vstat(x,y) = (v1_stat(x,y), v2_stat(x,y))
vstat_x_face = np.full((Ny, Nx - 1), 0.42)
vstat_y_face = np.zeros((Ny - 1, Nx))
# print(vstat_x.shape)
# print(vstat_y.shape)
# print(vstat_x.min(), vstat_x.max())
# print(vstat_y.min(), vstat_y.max())