import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from parameters import x_min, x_max, y_min, y_max

# Cartella del progetto
cartella = Path(__file__).resolve().parent

# Risultati della simulazione già completata
file_risultati = (
    cartella
    / "soluzioni_deviatore_tangente_con_nonlocale_2000x500.npz"
)

# Cartella in cui salvare i grafici
cartella_grafici = cartella / "grafici_deviatore_2000x500"
cartella_grafici.mkdir(exist_ok=True)

# Tempi fisici corrispondenti ai checkpoint salvati
tempi_reali = [0.01, 0.1, 0.5, 1.0, 1.5]

with np.load(file_risultati, allow_pickle=False) as dati:

    checkpoint = sorted(dati.files, key=int)
    print("Checkpoint disponibili:", checkpoint)

    if len(checkpoint) != len(tempi_reali):
        raise ValueError(
            f"Numero di checkpoint ({len(checkpoint)}) diverso "
            f"dal numero di tempi reali forniti ({len(tempi_reali)})."
        )

    for k, t in zip(checkpoint, tempi_reali):

        rho = dati[k]

        fig, ax = plt.subplots(figsize=(11, 5))

        im = ax.imshow(
            rho,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            aspect="equal",
            vmin=0,
            cmap="viridis"
        )

        # Segmento del deviatore
        ax.plot(
            [0.6, 1.9],
            [2.0, 0.7],
            "r--",
            linewidth=2,
            label="Deviatore"
        )

        ax.set_xlabel("Posizione longitudinale x [m]")
        ax.set_ylabel("Posizione trasversale y [m]")
        ax.set_title(f"t = {t:.4f} s")

        ax.legend()
        fig.colorbar(im, ax=ax, label="Densità")

        nome = f"densita_t_{t:.4f}.png"

        fig.savefig(
            cartella_grafici / nome,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"t = {t:.4f} s | "
            f"min = {rho.min():.6f} | "
            f"max = {rho.max():.6f} | "
            f"grafico salvato: {nome}",
            flush=True
        )

plt.show()