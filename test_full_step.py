import numpy as np

from initial_data import rho0
from nonlocal_term import compute_nonlocal_term, interfaces_interp
from lax_friedrichs import x_step, y_step
from cfl import lambda_x, lambda_y
from parameters import delta_x, delta_y


# Termine non locale calcolato a partire da rho^n
J1, J2 = compute_nonlocal_term(rho0)

# Valori del termine non locale sulle interfacce
J1_xface, J2_yface = interfaces_interp(J1, J2)

# Primo fractional step: direzione x
rho_half = x_step(rho0, J1_xface, lambda_x)

# Secondo fractional step: direzione y
rho_new = y_step(rho_half, J2_yface, lambda_y)


""" # Controlli
print("shape rho0:", rho0.shape)
print("shape rho_half:", rho_half.shape)
print("shape rho_new:", rho_new.shape)

print("min rho0:", np.min(rho0))
print("max rho0:", np.max(rho0))

print("min rho_half:", np.min(rho_half))
print("max rho_half:", np.max(rho_half))

print("min rho_new:", np.min(rho_new))
print("max rho_new:", np.max(rho_new))


mass_initial = np.sum(rho0) * delta_x * delta_y
mass_half = np.sum(rho_half) * delta_x * delta_y
mass_final = np.sum(rho_new) * delta_x * delta_y

print("massa iniziale:", mass_initial)
print("massa dopo x-step:", mass_half)
print("massa dopo y-step:", mass_final)

print("differenza massa x-step:", mass_half - mass_initial)
print("differenza massa totale:", mass_final - mass_initial)

tol = 1e-12

assert np.min(rho_half) >= -tol
assert np.min(rho_new) >= -tol

from cfl import lambda_x_max, lambda_y_max

assert lambda_x <= lambda_x_max
assert lambda_y <= lambda_y_max """