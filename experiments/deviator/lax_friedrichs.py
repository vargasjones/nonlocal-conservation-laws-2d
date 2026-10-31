import numpy as np

from parameters import eps, Nx, Ny
from heaviside import f, compute_Lf
from velocity_field import vstat_x_face, vstat_y_face

Lf = compute_Lf()
alpha = np.max(np.abs(vstat_x_face)) + eps * Lf
beta = np.max(np.abs(vstat_y_face)) + eps * Lf

def F(vstat_x_face, J1_xface, u, w):
    return 0.5 * (vstat_x_face * (u + w) + J1_xface * (f(u) + f(w))) - 0.5 * alpha * (w - u)

def G(vstat_y_face, J2_yface, u, w):
    return 0.5 * (vstat_y_face * (u + w) + J2_yface * (f(u) + f(w))) - 0.5 * beta * (w - u)

def x_step(rho, J1_xface, lambda_x):
    rho_half = rho.copy()

    # Calcolo dei flussi su tutte le interfacce interne
    F_faces = F(
        vstat_x_face,
        J1_xface,
        rho[:, :-1],
        rho[:, 1:]
    )

    # Celle interne
    rho_half[:, 1:-1] = rho[:, 1:-1] - lambda_x * (
        F_faces[:, 1:] - F_faces[:, :-1]
    )

    # Bordo sinistro: F_left = 0
    rho_half[:, 0] = rho[:, 0] - lambda_x * F_faces[:, 0]

    # Bordo destro: F_right = 0
    rho_half[:, -1] = rho[:, -1] + lambda_x * F_faces[:, -1]

    return rho_half

def y_step(rho_half, J2_yface, lambda_y):
    rho_new = rho_half.copy()

    # Calcolo dei flussi su tutte le interfacce interne
    G_faces = G(
        vstat_y_face,
        J2_yface,
        rho_half[:-1, :],
        rho_half[1:, :]
    )

    # Celle interne
    rho_new[1:-1, :] = rho_half[1:-1, :] - lambda_y * (
        G_faces[1:, :] - G_faces[:-1, :]
    )

    # Bordo inferiore: G_down = 0
    rho_new[0, :] = rho_half[0, :] - lambda_y * G_faces[0, :]

    # Bordo superiore: G_up = 0
    rho_new[-1, :] = rho_half[-1, :] + lambda_y * G_faces[-1, :]

    return rho_new