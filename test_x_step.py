import numpy as np

from initial_data import rho0
from nonlocal_term import compute_nonlocal_term, interfaces_interp
from lax_friedrichs import x_step
from cfl import lambda_x
from parameters import delta_x, delta_y


# 1. Calcolo del termine non locale nei centri
J1, J2 = compute_nonlocal_term(rho0)

# 2. Interpolazione alle interfacce
J1_xface, J2_yface = interfaces_interp(J1, J2)

# 3. Primo step di Lax-Friedrichs in direzione x
rho_half = x_step(rho0, J1_xface, lambda_x)


# 4. Controlli
""" print("shape rho_half:", rho_half.shape)

print("min rho_half:", np.min(rho_half))
print("max rho_half:", np.max(rho_half))

mass_before = np.sum(rho0) * delta_x * delta_y
mass_after = np.sum(rho_half) * delta_x * delta_y

print("massa prima:", mass_before)
print("massa dopo x-step:", mass_after)
print("differenza massa:", mass_after - mass_before) """