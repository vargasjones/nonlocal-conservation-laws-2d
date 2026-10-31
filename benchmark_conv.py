from time import perf_counter

from initial_data import rho0
from nonlocal_term import compute_nonlocal_term
from cfl import delta_t

T = 1.0
N_T = int(T / delta_t)

rho = rho0.copy()

# Prima chiamata di riscaldamento
compute_nonlocal_term(rho)

prove = 10

inizio = perf_counter()

for _ in range(prove):
    J1, J2 = compute_nonlocal_term(rho)

tempo_medio = (perf_counter() - inizio) / prove

tempo_extra = N_T * tempo_medio
tempo_totale = 27 * 60 + tempo_extra

print(f"Iterazioni totali: {N_T}")
print(f"Tempo medio convoluzione: {tempo_medio:.4f} s")
print(f"Tempo aggiuntivo stimato: {tempo_extra / 60:.2f} minuti")
print(f"Tempo complessivo stimato: {tempo_totale / 60:.2f} minuti")