import hashlib
import json
import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from parameters import Nx, Ny
from lax_friedrichs import x_step, y_step
from cfl import delta_t, delta_x, delta_y, lambda_x, lambda_y
from initial_data import rho0


# Identici tempi e parametri numerici della simulazione completa.
T = 1.5
tempi = [0.01, 0.1, 0.5, 1.0, 1.5]
N_T = int(T / delta_t)
indici = {int(t / delta_t) for t in tempi}
INTERVALLO_CHECKPOINT = 250
MODELLO = "senza_nonlocale"

if (Nx, Ny) != (2000, 500):
    raise RuntimeError(
        f"Griglia errata: Nx={Nx}, Ny={Ny}; attesa 2000 x 500. "
        "Verificare parameters.py prima di avviare il calcolo."
    )
if rho0.shape != (Ny, Nx):
    raise RuntimeError(f"rho0 ha forma {rho0.shape}; attesa {(Ny, Nx)}")
if len(indici) != len(tempi) or min(indici) < 1 or max(indici) > N_T:
    raise RuntimeError("Gli istanti richiesti non sono compatibili con il passo temporale.")

cartella_progetto = Path(__file__).resolve().parent
cartella_checkpoint = cartella_progetto / "checkpoint_deviatore_senza_nonlocale_2000x500"
file_risultati = (
    cartella_progetto / "soluzioni_deviatore_tangente_senza_nonlocale_2000x500.npz"
)
file_tempi = cartella_progetto / "tempi_deviatore_senza_nonlocale_2000x500.json"

# La firma riguarda i moduli che stabiliscono il problema e lo schema numerico.
# E' calcolata sullo stesso insieme di file della simulazione completa.
file_modello = (
    "parameters.py", "grid.py", "cfl.py", "initial_data.py",
    "velocity_field.py", "kernel.py", "heaviside.py",
    "nonlocal_term.py", "lax_friedrichs.py",
)


def firma_modello():
    h = hashlib.sha256()
    for nome in file_modello:
        p = cartella_progetto / nome
        h.update(nome.encode("utf-8"))
        if p.is_file():
            h.update(p.read_bytes())
        else:
            h.update(b"<file assente>")
    h.update(np.ascontiguousarray(rho0).tobytes())
    return h.hexdigest()


def salva_npz_atomicamente(destinazione, contenuto):
    """Pubblica il file .npz solo dopo averlo scritto completamente."""
    temporaneo = destinazione.with_name(destinazione.name + ".part")
    with temporaneo.open("wb") as f:
        np.savez(f, **contenuto)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporaneo, destinazione)


firma_corrente = firma_modello()

# Controllo, ove disponibile, della configurazione dell'esecuzione completa.
# Non legge le grandi matrici del run con termine non locale.
checkpoint_riferimento = (
    cartella_progetto
    / "checkpoint_deviatore_2000x500"
    / f"checkpoint_{N_T:06d}.npz"
)
if checkpoint_riferimento.is_file():
    with np.load(checkpoint_riferimento, allow_pickle=False) as precedente:
        for chiave, atteso in (
            ("delta_t", delta_t),
            ("delta_x", delta_x),
            ("delta_y", delta_y),
            ("N_T", N_T),
        ):
            if chiave in precedente and precedente[chiave].item() != atteso:
                raise RuntimeError(
                    f"Il parametro {chiave} non coincide con il calcolo "
                    "con termine non locale. Non avvio una simulazione non confrontabile."
                )
        if "firma_modello" in precedente:
            firma_precedente = str(precedente["firma_modello"].item())
            if firma_precedente != firma_corrente:
                raise RuntimeError(
                    "I file del modello differiscono da quelli del run con "
                    "termine non locale. Ripristinare i file originali."
                )
    print("Configurazione confrontata con il checkpoint del run completo: OK", flush=True)
else:
    print(
        "Checkpoint finale del run completo non trovato: controllo incrociato "
        "non disponibile. Verificare che parameters.py e gli altri moduli "
        "siano gli stessi del run con termine non locale.",
        flush=True,
    )

cartella_checkpoint.mkdir(parents=True, exist_ok=True)

rho = rho0.copy()
soluzioni = {}
n_inizio = 0

for file in sorted(cartella_checkpoint.glob("checkpoint_*.npz"), reverse=True):
    try:
        with np.load(file, allow_pickle=False) as dati:
            passo = int(dati["passo"])
            rho_salvata = dati["rho"].copy()
            dt_salvato = float(dati["delta_t"])
            nt_salvato = int(dati["N_T"])
            dx_salvato = float(dati["delta_x"])
            dy_salvato = float(dati["delta_y"])
            modello_salvato = str(dati["modello"].item())
            firma_salvata = str(dati["firma_modello"].item())
            soluzioni_salvate = {
                int(nome[4:]): dati[nome].copy()
                for nome in dati.files if nome.startswith("sol_")
            }
    except (OSError, ValueError, EOFError, KeyError, BadZipFile):
        print(f"Checkpoint incompleto o non leggibile: {file.name}", flush=True)
        continue

    if (
        rho_salvata.shape != (Ny, Nx)
        or dt_salvato != delta_t
        or nt_salvato != N_T
        or dx_salvato != delta_x
        or dy_salvato != delta_y
        or modello_salvato != MODELLO
        or firma_salvata != firma_corrente
        or not (0 <= passo <= N_T)
        or any(
            k not in indici or k > passo or v.shape != (Ny, Nx)
            for k, v in soluzioni_salvate.items()
        )
        or any(k <= passo and k not in soluzioni_salvate for k in indici)
    ):
        raise RuntimeError(
            f"Checkpoint {file.name} incompatibile con codice, configurazione "
            "o modello attuali. Non riparto automaticamente."
        )

    rho = rho_salvata
    soluzioni = soluzioni_salvate
    n_inizio = passo
    print(f"Ripresa dal passo {n_inizio}/{N_T}", flush=True)
    break

print(f"Modello: {MODELLO}", flush=True)
print(f"Griglia: {Nx} x {Ny}; dx={delta_x:.12g}; dy={delta_y:.12g}", flush=True)
print(f"dt={delta_t:.16g}; passi totali={N_T}", flush=True)
print(f"Passi di osservazione: {sorted(indici)}", flush=True)
print(f"Cartella checkpoint: {cartella_checkpoint}", flush=True)

# Nessun ricalcolo della convoluzione: qui J1 = J2 = 0 in tutte le celle.
# I coefficienti lambda_x, lambda_y provengono dallo stesso modulo cfl.py
# usato nella simulazione con termine non locale.
J1_xface = np.zeros((Ny, Nx - 1), dtype=rho.dtype)
J2_yface = np.zeros((Ny - 1, Nx), dtype=rho.dtype)

for n in range(n_inizio, N_T):
    rho_half = x_step(rho, J1_xface, lambda_x)
    rho_new = y_step(rho_half, J2_yface, lambda_y)
    rho = rho_new

    if n + 1 in indici:
        soluzioni[n + 1] = rho.copy()

    if (n + 1) % 100 == 0:
        print(f"Iterazione {n + 1}/{N_T}", flush=True)

    if (n + 1) % INTERVALLO_CHECKPOINT == 0 or (n + 1) == N_T:
        passo_corrente = n + 1
        dati_checkpoint = {
            "passo": passo_corrente,
            "rho": rho,
            "delta_t": delta_t,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "N_T": N_T,
            "modello": np.array(MODELLO),
            "firma_modello": np.array(firma_corrente),
        }
        dati_checkpoint.update({f"sol_{k}": v for k, v in soluzioni.items()})
        percorso = cartella_checkpoint / f"checkpoint_{passo_corrente:06d}.npz"
        salva_npz_atomicamente(percorso, dati_checkpoint)
        print(f"Checkpoint salvato al passo {passo_corrente}", flush=True)

salva_npz_atomicamente(file_risultati, {str(k): v for k, v in soluzioni.items()})

# Tempi effettivi e parametri disponibili separatamente: il file delle
# soluzioni mantiene soltanto le cinque chiavi numeriche, come quello completo.
metadati = {
    "modello": MODELLO,
    "Nx": Nx,
    "Ny": Ny,
    "delta_x": float(delta_x),
    "delta_y": float(delta_y),
    "delta_t": float(delta_t),
    "N_T": N_T,
    "tempi_nominali": tempi,
    "passi_osservati": sorted(soluzioni),
    "tempi_effettivi": {str(k): float(k * delta_t) for k in sorted(soluzioni)},
}
file_tempi.write_text(json.dumps(metadati, indent=2), encoding="utf-8")

massa_iniziale = np.sum(rho0) * delta_x * delta_y
massa_finale = np.sum(rho) * delta_x * delta_y
print("\nSIMULAZIONE COMPLETATA", flush=True)
print(f"Massa iniziale: {massa_iniziale:.12f}")
print(f"Massa finale:   {massa_finale:.12f}")
print(f"Differenza:    {massa_finale - massa_iniziale:.3e}")
print(f"Norma infinito finale: {np.max(rho):.12f}")
print(f"Soluzioni salvate in: {file_risultati}")
print(f"Tempi effettivi salvati in: {file_tempi}")
