import numpy as np
from parameters import Nx, Ny, x_min, x_max, y_min, y_max, delta_x, delta_y

x = np.zeros(Nx)
y = np.zeros(Ny)

for i in range(Nx):
    x[i] = x_min + (i + 1/2) * delta_x

for j in range(Ny):
    y[j] = y_min + (j + 1/2) * delta_y


X, Y = np.meshgrid(x, y)
# print(X.shape)
# print(Y.shape)