"""Analisi delle soluzioni numeriche dell'esperimento con deviatore.

Calcola massa, estremi, variazione totale discreta e massa vicino a P.
Utilizza il file delle soluzioni a risoluzione 2000x500 oppure gli stati
contenuti nei checkpoint disponibili. I checkpoint forniscono, quando
presenti, i tempi effettivi e le metriche degli stati intermedi.
"""

import csv
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

CARTELLA = Path(__file__).resolve().parent
RISULTATI = CARTELLA / "soluzioni_deviatore_tangente_con_nonlocale_2000x500.npz"
DIR_CHECKPOINT = CARTELLA / "checkpoint_deviatore_2000x500"

NX, NY = 2000, 500
X_MIN, X_MAX = 0.0, 4.0
Y_MIN, Y_MAX = 0.0, 2.0
DX = (X_MAX - X_MIN) / NX
DY = (Y_MAX - Y_MIN) / NY
MASSA_INIZIALE = 0.56
TEMPI_NOMINALI = [0.01, 0.1, 0.5, 1.0, 1.5]

# Regione geometrica prefissata, indipendente dai risultati della simulazione.
P_X, P_Y = 1.9, 0.7
RAGGIO_P = 0.20  # metri
x_centri = X_MIN + (np.arange(NX) + 0.5) * DX
y_centri = Y_MIN + (np.arange(NY) + 0.5) * DY
REGIONE_P = ((x_centri[None, :] - P_X)**2 +
             (y_centri[:, None] - P_Y)**2 <= RAGGIO_P**2)


def metriche(rho):
    if rho.shape != (NY, NX):
        raise ValueError(f"Matrice di forma {rho.shape}: attesa ({NY}, {NX}).")
    if not np.isfinite(rho).all():
        raise ValueError("La matrice contiene NaN o infiniti.")

    massa = float(np.sum(rho, dtype=np.float64) * DX * DY)
    # TV della ricostruzione costante a tratti, prolungata per zero
    # all'esterno del dominio, coerente con le differenze su Z^2.
    tv_x = DY * (
        np.abs(np.diff(rho, axis=1)).sum(dtype=np.float64)
        + np.abs(rho[:, 0]).sum(dtype=np.float64)
        + np.abs(rho[:, -1]).sum(dtype=np.float64)
    )
    tv_y = DX * (
        np.abs(np.diff(rho, axis=0)).sum(dtype=np.float64)
        + np.abs(rho[0, :]).sum(dtype=np.float64)
        + np.abs(rho[-1, :]).sum(dtype=np.float64)
    )
    massa_p = float(np.sum(rho[REGIONE_P], dtype=np.float64) * DX * DY)

    return {
        "massa": massa,
        "errore_massa": massa - MASSA_INIZIALE,
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "TV_x": float(tv_x),
        "TV_y": float(tv_y),
        "TV_totale": float(tv_x + tv_y),
        "massa_vicino_P": massa_p,
        "percentuale_massa_vicino_P": 100.0 * massa_p / massa if massa else float('nan'),
    }


def salva_csv(percorso, righe):
    if not righe:
        return
    with percorso.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=righe[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(righe)
    print(f"CSV salvato: {percorso}")


# Si cercano i file con checkpoint sia nella sottocartella sia accanto allo script.
files_ckpt = sorted(
    {*DIR_CHECKPOINT.glob("checkpoint_*.npz"), *CARTELLA.glob("checkpoint_*.npz")},
    key=lambda p: p.name,
    reverse=True,
)
DT_HORIZON = None
N_T_HORIZON = None
ckpt_valido = None
for f in files_ckpt:
    try:
        with np.load(f, allow_pickle=False) as d:
            dt = float(d["delta_t"])
            nt = int(d["N_T"])
            passo = int(d["passo"])
            if dt <= 0 or not (0 <= passo <= nt):
                continue
            if "delta_x" in d.files and not np.isclose(float(d["delta_x"]), DX):
                continue
            if "delta_y" in d.files and not np.isclose(float(d["delta_y"]), DY):
                continue
            if d["rho"].shape != (NY, NX):
                continue
        DT_HORIZON, N_T_HORIZON, ckpt_valido = dt, nt, f
        break
    except (OSError, ValueError, EOFError, KeyError, BadZipFile):
        print(f"Checkpoint ignorato perché non leggibile: {f.name}")

if DT_HORIZON is None:
    print("Nessun checkpoint completo trovato: tempi fisici esatti non disponibili.")
    print("I tempi NOMINALI saranno mostrati separatamente; nessun dt locale viene utilizzato.")
else:
    print(f"Metadati letti da: {ckpt_valido.name}")
    print(f"dt Horizon = {DT_HORIZON:.15g} s; N_T Horizon = {N_T_HORIZON}")


# Preferisce il file finale delle cinque soluzioni; se assente,
# può usare i sol_* dell'ultimo checkpoint completo.
if RISULTATI.is_file():
    sorgente = RISULTATI
    prefisso = ""
elif ckpt_valido is not None:
    sorgente = ckpt_valido
    prefisso = "sol_"
else:
    raise FileNotFoundError(
        "Servono il file finale delle soluzioni oppure l'ultimo checkpoint completo."
    )

righe = []
with np.load(sorgente, allow_pickle=False) as dati:
    keys = sorted(
        (k for k in dati.files if k.startswith(prefisso) and k[len(prefisso):].isdigit()),
        key=lambda k: int(k[len(prefisso):])
    )
    if len(keys) != len(TEMPI_NOMINALI):
        raise ValueError(
            f"Trovate {len(keys)} soluzioni invece delle {len(TEMPI_NOMINALI)} previste: {keys}"
        )
    print("\nRISULTATI AI CINQUE ISTANTI")
    print("passo | tempo nominale | tempo effettivo | massa | max rho | TV | massa in B(P,0.20)")
    for nome, nominale in zip(keys, TEMPI_NOMINALI):
        passo = int(nome[len(prefisso):])
        t_eff = passo * DT_HORIZON if DT_HORIZON is not None else None
        vals = metriche(dati[nome])
        riga = {"passo": passo, "tempo_nominale_s": nominale,
                 "tempo_effettivo_s": t_eff if t_eff is not None else "", **vals}
        righe.append(riga)
        print(f"{passo:5d} | {nominale:13.2f} | "
              f"{(f'{t_eff:.6f}' if t_eff is not None else 'non disponibile'):>14} | "
              f"{vals['massa']:.10f} | {vals['rho_max']:.6f} | "
              f"{vals['TV_totale']:.6f} | {vals['massa_vicino_P']:.8f}")

salva_csv(CARTELLA / "metriche_cinque_istanti_2000x500.csv", righe)

# Facoltativo: se sono presenti *tutti* i checkpoint periodici, si producono
# anche le metriche per ognuno, senza riavviare alcuna simulazione.
righe_intermedie = []
if files_ckpt and DT_HORIZON is not None:
    print("\nAnalisi degli stati nei checkpoint disponibili...")
    for f in sorted(files_ckpt, key=lambda p: p.name):
        try:
            with np.load(f, allow_pickle=False) as d:
                passo = int(d["passo"])
                if (float(d["delta_t"]) != DT_HORIZON or
                        int(d["N_T"]) != N_T_HORIZON):
                    print(f"Checkpoint con tempi incompatibili ignorato: {f.name}")
                    continue
                vals = metriche(d["rho"])
            righe_intermedie.append({"passo": passo,
                                     "tempo_effettivo_s": passo * DT_HORIZON,
                                     **vals})
        except (OSError, ValueError, EOFError, KeyError, BadZipFile) as e:
            print(f"Checkpoint ignorato: {f.name} ({e})")

    salva_csv(CARTELLA / "metriche_checkpoint_intermedi_2000x500.csv", righe_intermedie)

print("\nAnalisi completata: nessun passo di evoluzione numerica è stato eseguito.")
