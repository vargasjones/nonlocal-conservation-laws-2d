import numpy as np

from velocity_field import B
from parameters import delta_x, delta_y
from cfl import delta_t

dati = np.load("soluzioni_deviatore_tangente_con_nonlocale.npz")

for k in sorted(dati.files, key=int):

    rho = dati[k]

    massa_totale = np.sum(rho) * delta_x * delta_y
    massa_B = np.sum(rho[B]) * delta_x * delta_y

    percentuale = 100 * massa_B / massa_totale

    print(f"t = {int(k) * delta_t:.4f} s")
    print(f"Massa in B: {massa_B:.8f}")
    print(f"Percentuale: {percentuale:.4f}%")
    print()
