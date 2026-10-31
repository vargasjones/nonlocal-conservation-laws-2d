import numpy as np
from parameters import rho_max

def H_atan(rho):
    return (1/np.pi) * np.atan (50 * (rho - rho_max) ) + 1/2

def f(rho):
    return rho * H_atan(rho)

def compute_Lf():
    rho_values = np.linspace(0, 2*rho_max, 100000)

    H_values = H_atan(rho_values)

    dH_values = np.gradient(H_values, rho_values)

    df_values = H_values + rho_values * dH_values

    Lf = np.max(np.abs(df_values))

    return Lf
# print(H_atan(0))
# print(H_atan(1))
# print(H_atan(2))
# print(f(0))
# print(f(1))
# print(f(2))