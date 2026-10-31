
import numpy as np
import matplotlib.pyplot as plt

from parameters import (
    Nx, Ny, x_min, x_max, y_min, y_max,
    delta_x, delta_y, v_T, xP, yP, xQ, yQ, d, d_B
)
from grid import x, y


# Geometria del deviatore

s0 = xP + yP
r_min = xQ - yQ
r_max = xP - yP

sqrt2 = np.sqrt(2)


# REGIONE B

B = np.zeros((Ny, Nx), dtype=bool)

for i in range(Nx):
    for j in range(Ny):

        if (s0 < x[i] + y[j] <= s0 + d_B * sqrt2
                and r_min <= x[i] - y[j] <= r_max):

            B[j, i] = True


# REGIONE C

C = np.zeros((Ny, Nx), dtype=bool)

for i in range(Nx):
    for j in range(Ny):

        if (s0 - d * sqrt2 <= x[i] + y[j] <= s0
                and r_min <= x[i] - y[j] <= r_max):

            C[j, i] = True


# REGIONE A

A = ~(B | C)


# PARAMETRI DEL NUOVO CAMPO STATICO

x_c = 1.35
larghezza_x = 0.38

y_c = 1.0
sigma_y = 0.6

spostamento = 0.9


# CAMPO DI VELOCITA' REGOLARE


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t**3 * (10 - 15*t + 6*t**2)


def campo_velocita(xx, yy):

    # Campo regolare precedente

    z = (xx - x_c) / larghezza_x

    S = 0.5 * (1 + np.tanh(z))
    S_x = (1 - np.tanh(z)**2) / (2 * larghezza_x)

    g = np.exp(-0.5 * ((yy - y_c) / sigma_y)**2)
    g_y = -(yy - y_c) / sigma_y**2 * g

    vx = v_T * (1 + spostamento * S * g_y)
    vy = -v_T * spostamento * S_x * g


    # Correzione vicino al deviatore

    q = xx + yy - s0
    r = xx - yy

    L_n = 0.5
    L_e = 0.5

    peso = np.exp(-(q / L_n)**2)

    peso *= smoothstep((r - r_min + L_e) / L_e)
    peso *= smoothstep((r_max + L_e - r) / L_e)


    # Eliminazione graduale della componente normale

    correzione = 0.5 * peso * (vx + vy)

    vx = vx - correzione
    vy = vy - correzione

    return vx, vy


# VELOCITA' SULLE INTERFACCE VERTICALI

vstat_x_face = np.zeros((Ny, Nx - 1))

for i in range(Nx - 1):
    for j in range(Ny):

        xx = x_min + (i + 1) * delta_x
        yy = y[j]

        vx, vy = campo_velocita(xx, yy)

        vstat_x_face[j, i] = vx


# VELOCITA' SULLE INTERFACCE ORIZZONTALI

vstat_y_face = np.zeros((Ny - 1, Nx))

for i in range(Nx):
    for j in range(Ny - 1):

        xx = x[i]
        yy = y_min + (j + 1) * delta_y

        vx, vy = campo_velocita(xx, yy)

        vstat_y_face[j, i] = vy


# VISUALIZZAZIONE DEL CAMPO

if __name__ == "__main__":

    regioni = np.zeros((Ny, Nx), dtype=int)

    regioni[B] = 1
    regioni[C] = 2

    plt.figure()

    plt.imshow(
        regioni,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="equal",
        cmap="viridis",
        vmin=0,
        vmax=2,
        alpha=0.5
    )

    plt.plot(
        [xQ, xP],
        [yQ, yP],
        "r--",
        linewidth=2,
        label="Deviatore"
    )

    # Frecce del campo di velocita'

    X, Y = np.meshgrid(x[::20], y[::20])

    U, V = campo_velocita(X, Y)

    plt.quiver(X, Y, U, V, color="white")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Campo di velocita' statico")
    plt.legend()
    plt.show()
    print("\nControllo tangenza:")

    for xx in np.linspace(xQ, xP, 10):

        yy = s0 - xx

        vx, vy = campo_velocita(xx, yy)

        print(f"x = {xx:.3f}, v1 + v2 = {vx + vy:.3e}")