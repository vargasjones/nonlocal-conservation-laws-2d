import numpy as np

from parameters import delta_x, delta_y
from cfl import delta_t

senza = np.load("soluzioni_deviatore_tangente.npz")
con = np.load("soluzioni_deviatore_tangente_con_nonlocale.npz")

for k in sorted(set(senza.files) & set(con.files), key=int):

    rho_senza = senza[k]
    rho_con = con[k]

    differenza = rho_con - rho_senza

    massa = np.sum(rho_con) * delta_x * delta_y

    print(f"\nt = {int(k) * delta_t:.4f} s")

    print("Massa con nonlocale:", massa)
    print("Densità minima:", np.min(rho_con))

    print("Norma infinito senza:", np.max(np.abs(rho_senza)))
    print("Norma infinito con:", np.max(np.abs(rho_con)))

    print("Norma L1 della differenza:",
          np.sum(np.abs(differenza)) * delta_x * delta_y)

    print("Norma infinito della differenza:",
          np.max(np.abs(differenza)))