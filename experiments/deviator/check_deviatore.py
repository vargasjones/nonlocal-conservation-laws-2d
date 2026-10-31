import numpy as np

from initial_data import rho0
from parameters import delta_x, delta_y
from grid import x, y
dati = np.load("soluzioni_deviatore_tangente.npz")

massa_iniziale = np.sum(rho0) * delta_x * delta_y

print("Massa iniziale:", massa_iniziale)

for k in sorted(dati.files, key=int):

    rho = dati[k]

    massa = np.sum(rho) * delta_x * delta_y
    differenza = massa - massa_iniziale
    j, i = np.unravel_index(np.argmax(rho), rho.shape)

    print("Massimo in x =", x[i])
    print("Massimo in y =", y[j])
    print(f"\nPasso {k}")
    print("Massa:", massa)
    print("Differenza di massa:", differenza)
    print("Densità minima:", np.min(rho))
    print("Norma infinito:", np.max(np.abs(rho)))