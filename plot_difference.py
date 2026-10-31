import numpy as np
import matplotlib.pyplot as plt

from parameters import Nx, Ny
from cfl import delta_t, delta_x, delta_y

# Caricamento delle due simulazioni
nonloc = np.load("soluzioni.npz")
loc = np.load("soluzioni_senza_nonlocale.npz")

# Indici temporali comuni
indici = sorted(set(nonloc.files) & set(loc.files), key=int)

# Scala cromatica comune ai quattro grafici
limite = max(
    np.max(np.abs(nonloc[k] - loc[k]))
    for k in indici
)

if limite == 0:
    limite = 1e-12

for k in indici:
    D = nonloc[k] - loc[k]

    plt.figure()
    plt.imshow(
        D,
        origin="lower",
        extent=(0, Nx * delta_x, 0, Ny * delta_y),
        aspect="equal",
        cmap="RdBu_r",
        vmin=-limite,
        vmax=limite
    )

    plt.xlabel("Posizione longitudinale x [m]")
    plt.ylabel("Posizione trasversale y [m]")
    plt.title(f"Differenza delle densità a t = {int(k) * delta_t:.4f} s")
    plt.colorbar(label="Differenza di densità")
    plt.show()