import numpy as np
from scipy.signal import oaconvolve
from kernel import dx_eta, dy_eta , eta
from parameters import delta_x, delta_y, eps, Nx, Ny
from initial_data import rho0

def compute_nonlocal_term(rho):
    Cx = ( delta_x * delta_y) * oaconvolve(rho, dx_eta, mode="same") 
    Cy = ( delta_x * delta_y) * oaconvolve(rho, dy_eta, mode="same")
    den = np.sqrt(1 + Cx**2 + Cy**2)
    J1 = - eps * (Cx / den)
    J2 = - eps * (Cy / den)

    # print(Cx.min(), Cx.max())
    # print(Cy.min(), Cy.max())
    # print(Cx.shape, Cy.shape)
    # print(np.isnan(Cx).any(), np.isnan(Cy).any())
    # print(np.isinf(Cx).any(), np.isinf(Cy).any())

    # print(delta_x * delta_y * np.sum(eta))
    # eta è un kernel di convoluzione normalizzato: il suo integrale su R^2 deve valere 1.
    # In questo modo eta*rho rappresenta una media spaziale della densità senza alterarne artificialmente la scala.

    # print(delta_x * delta_y * np.sum(dx_eta))
    # print(delta_x * delta_y * np.sum(dy_eta))
    # Poiché eta è gaussiana e tende a 0 all'infinito, l'integrale su R^2 delle sue derivate parziali è nullo.
    # Equivalentemente, dx_eta è dispari in x e dy_eta è dispari in y, quindi i contributi positivi e negativi si cancellano.

    return J1, J2
  
def interfaces_interp(J1, J2):
    J1_xface = np.zeros((Ny, Nx - 1))
    J2_yface = np.zeros((Ny - 1, Nx))
    for j in range(Ny):
        for i in range(Nx - 1):
            J1_xface[j, i] = 0.5 * (J1[j, i] + J1[j, i+1])  

    for j in range(Ny - 1):
        for i in range(Nx):
            J2_yface[j, i] = 0.5 * (J2[j, i] + J2[j+1, i])
    
    return J1_xface, J2_yface

J1, J2 = compute_nonlocal_term(rho0)

J1_xface, J2_yface = interfaces_interp(J1, J2)

""" print(J1_xface.shape)
print(J2_yface.shape) """
# print(J1.shape)
# print(J2.shape)
# print(J1.min(), J1.max())
# print(J2.min(), J2.max())
# I_norm = np.sqrt(J1**2 + J2**2) serve solo per controllare che la proprietà |I(rho)|<eps sia rispettata

# print(I_norm.min())
# print(I_norm.max())
# print(eps)