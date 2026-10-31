import numpy as np
import matplotlib.pyplot as plt

from parameters import Nx, Ny
from lax_friedrichs import x_step, y_step
from cfl import delta_t, delta_x, delta_y
from initial_data import rho0
from cfl import lambda_x , lambda_y

tempi = [0.01, 0.1, 0.5, 1.0]
indici = {int(t / delta_t) for t in tempi}
soluzioni = {}
T = 1.0
N_T = int(T / delta_t)
rho = rho0
for n in range(N_T):
    J1_xface = np.zeros((Ny, Nx - 1))
    J2_yface = np.zeros((Ny - 1, Nx))
    rho_half = x_step(rho, J1_xface, lambda_x)
    rho_new = y_step(rho_half, J2_yface, lambda_y)
    rho = rho_new
    if n + 1 in indici:
        soluzioni[n + 1] = rho.copy()
    if (n + 1) % 100 == 0:
        print(f"Iterazione {n+1}/{N_T}", flush=True)
np.savez("soluzioni_senza_nonlocale.npz",
         **{str(k): v for k, v in soluzioni.items()})
 # conservazione della massa



for k, rho_k in soluzioni.items():
    plt.figure()
    plt.imshow(
        rho_k,
        origin="lower",
        extent=(0, Nx * delta_x, 0, Ny * delta_y),
        aspect="equal",
        vmin=0,
        vmax=1
    )
    plt.xlabel("Posizione longitudinale x [m]")
    plt.ylabel("Posizione trasversale y [m]")
    plt.title(f"t = {k * delta_t:.4f} s")
    plt.colorbar(label="Densità")
    plt.show()

    