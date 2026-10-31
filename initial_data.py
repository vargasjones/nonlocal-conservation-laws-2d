import numpy as np
import matplotlib.pyplot as plt
from parameters import Nx, Ny
from grid import X, Y, x , y
rho0 = np.zeros((Ny, Nx))



a = 2
b = 8
c = 1
d = 4

for i in range(Nx):
    for j in range(Ny):
        if a <= x[i] <= b and c <= y[j] <= d:
            rho0[j, i] = 1
# print(rho0.shape)
# print(np.max(rho0))
# print(np.min(rho0))
# plt.pcolormesh(X, Y, rho0, shading="auto")
# plt.colorbar(label="rho_0")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Dato iniziale")
# plt.show()

