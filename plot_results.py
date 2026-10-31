import numpy as np
import matplotlib.pyplot as plt

from parameters import Nx, Ny, delta_x, delta_y
from cfl import delta_t
from initial_data import rho0

dati = np.load("soluzioni.npz")

for k in dati.files:
    rho_k = dati[k]
    t = int(k) * delta_t

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
    plt.title(f"t = {t:.4f} s")
    plt.colorbar(label="Densità")

    plt.savefig(f"densita_t_{t:.4f}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

for k in sorted(dati.files, key=int):
    rho_norm = np.max(np.abs(dati[k]))
    print(f"t = {int(k)*delta_t:.4f} s, norma = {rho_norm:.6f}")



dati = np.load("soluzioni.npz")

massa_iniziale = np.sum(rho0) * delta_x * delta_y
print("Massa iniziale:", massa_iniziale)

for k in sorted(dati.files, key=int):
    massa = np.sum(dati[k]) * delta_x * delta_y
    print(f"Passo {k}: massa = {massa:.12f}")