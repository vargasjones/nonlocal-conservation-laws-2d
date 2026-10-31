import numpy as np

from parameters import delta_x, delta_y, xP, yP, xQ, yQ
from grid import X, Y
from velocity_field import vstat_x_face, vstat_y_face
from lax_friedrichs import F, G
from cfl import delta_t

dati = np.load("soluzioni_deviatore_tangente.npz")

# Retta del deviatore: x + y = s0

s0 = xP + yP
r_min = xQ - yQ
r_max = xP - yP

# Escludiamo le zone vicine alle estremità
margine = 0.2

# Taglio leggermente spostato dalla retta,
# per evitare che passi esattamente sui centri
s_cut = s0 + delta_x / 2

# Interfacce verticali attraversate dal taglio
mask_x = (
    (X[:, :-1] + Y[:, :-1] <= s_cut)
    & (X[:, 1:] + Y[:, 1:] > s_cut)
)

r_x = s_cut - 2 * Y[:, :-1]

mask_x &= (
    (r_x >= r_min + margine)
    & (r_x <= r_max - margine)
)

# Interfacce orizzontali attraversate dal taglio
mask_y = (
    (X[:-1, :] + Y[:-1, :] <= s_cut)
    & (X[1:, :] + Y[1:, :] > s_cut)
)

r_y = 2 * X[:-1, :] - s_cut

mask_y &= (
    (r_y >= r_min + margine)
    & (r_y <= r_max - margine)
)

for k in sorted(dati.files, key=int):

    rho = dati[k]

    J1 = np.zeros_like(vstat_x_face)
    J2 = np.zeros_like(vstat_y_face)

    F_faces = F(
        vstat_x_face, J1,
        rho[:, :-1], rho[:, 1:]
    )

    G_faces = G(
        vstat_y_face, J2,
        rho[:-1, :], rho[1:, :]
    )

    flusso_x = np.sum(F_faces[mask_x]) * delta_y
    flusso_y = np.sum(G_faces[mask_y]) * delta_x

    flusso_netto = flusso_x + flusso_y

    print(f"t = {int(k) * delta_t:.4f} s")
    print(f"Flusso numerico netto: {flusso_netto:.8e}")
    print()