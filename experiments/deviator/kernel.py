import numpy as np
from parameters import x_min, x_max, Nx, Ny, delta_x, delta_y
csi = np.arange(-(Nx - 1), Nx) * delta_x
zeta = np.arange(-(Ny - 1), Ny) * delta_y
CSI, ZETA = np.meshgrid(csi, zeta)
# print(CSI.shape)
# print(ZETA.shape)

sigma = 10000
eta = (sigma/(2*np.pi)) * np.exp((-sigma/2)*(CSI**2 + ZETA**2))
dx_eta = - sigma * CSI * eta
dy_eta = - sigma * ZETA * eta
# print(eta.shape)
# print(dx_eta.shape)
# print(dy_eta.shape)