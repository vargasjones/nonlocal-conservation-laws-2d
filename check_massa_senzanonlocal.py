import numpy as np

from initial_data import rho0
from parameters import delta_x, delta_y
from cfl import delta_t

# Caricamento delle soluzioni già calcolate
dati = np.load("soluzioni_senza_nonlocale.npz")

# Massa iniziale
mass_iniziale = np.sum(rho0) * delta_x * delta_y

print(f"Massa iniziale: {mass_iniziale:.12f}\n")

# Controllo della massa agli istanti salvati
for k in sorted(dati.files, key=int):

    rho = dati[k]

    mass_attuale = np.sum(rho) * delta_x * delta_y

    differenza = mass_attuale - mass_iniziale

    errore_relativo = abs(differenza) / abs(mass_iniziale)

    print(f"Tempo: {int(k) * delta_t:.4f} s")
    print(f"Massa: {mass_attuale:.12f}")
    print(f"Differenza dalla massa iniziale: {differenza:.3e}")
    print(f"Errore relativo: {errore_relativo:.3e}")
    print()
