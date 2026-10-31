import numpy as np

from initial_data import rho0
from nonlocal_term import compute_nonlocal_term, interfaces_interp
from lax_friedrichs import x_step, y_step
from cfl import lambda_x, lambda_y, delta_x, delta_y

rho = rho0.copy()

J1, J2 = compute_nonlocal_term(rho)
J1_xface, J2_yface = interfaces_interp(J1, J2)

rho_half = x_step(rho, J1_xface, lambda_x)
rho_new = y_step(rho_half, J2_yface, lambda_y)

mass_initial = np.sum(rho0) * delta_x * delta_y
mass_half = np.sum(rho_half) * delta_x * delta_y
mass_final = np.sum(rho_new) * delta_x * delta_y

print("Massa iniziale:", mass_initial)
print("Massa dopo x-step:", mass_half)
print("Massa dopo y-step:", mass_final)

print("Differenza di massa:", mass_final - mass_initial)
print("Densità minima:", np.min(rho_new))
print("Densità massima:", np.max(rho_new))

assert rho_new.shape == rho0.shape
assert np.min(rho_new) >= -1e-12