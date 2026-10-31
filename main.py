import numpy as np
import matplotlib.pyplot as plt

from parameters import Nx, Ny
from lax_friedrichs import x_step, y_step
from cfl import delta_t, delta_x, delta_y
from nonlocal_term import compute_nonlocal_term, interfaces_interp, J1_xface, J2_yface
from initial_data import rho0
from cfl import lambda_x , lambda_y

tempi = [0.01, 0.1, 0.5, 1.0]
indici = {int(t / delta_t) for t in tempi}
soluzioni = {}
T = 1.0
N_T = int(T / delta_t)
rho = rho0
for n in range(N_T):
    J1, J2 = compute_nonlocal_term(rho)
    J1_xface, J2_yface = interfaces_interp(J1, J2)
    rho_half = x_step(rho, J1_xface, lambda_x)
    rho_new = y_step(rho_half, J2_yface, lambda_y)
    rho = rho_new
    if n + 1 in indici:
        soluzioni[n + 1] = rho.copy()
    if (n + 1) % 100 == 0:
        print(f"Iterazione {n+1}/{N_T}", flush=True)
np.savez("soluzioni.npz", **{str(k): v for k, v in soluzioni.items()})
 # conservazione della massa


""" mass_initial = np.sum(rho0) * delta_x * delta_y
mass_half = np.sum(rho_half) * delta_x * delta_y
mass_final = np.sum(rho_new) * delta_x * delta_y """
rho_norm = np.max(rho) #ho già verificato la positività quindi posso togliere abs
""" print("massa iniziale:", mass_initial)
print("massa dopo x-step:", mass_half)
print("massa dopo y-step:", mass_final)
 """


"""
# positività

tol = 1e-12
assert np.min(rho_new) >= -tol

plt.imshow(rho0, origin='lower', aspect='auto')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Densità iniziale")
plt.colorbar(label="densità")
plt.show()

plt.imshow(rho, origin='lower', aspect='auto')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Densità iniziale")
plt.colorbar(label="densità")
plt.show() """

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

    