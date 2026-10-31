# Material transport with a deviator

*Second numerical experiment — two-dimensional nonlocal conservation law*

This experiment concerns the evolution of a localized material density on a conveyor belt with a deviator. I use a spatially varying prescribed velocity field to represent the deflection of material and compare the resulting density distributions in the presence and absence of the nonlocal interaction. Both simulations use the Lax–Friedrichs finite-volume method described in the [main README](../../README.md).

## Geometry and initial density

The computational domain is $\Omega=[0,4]\times[0,2]$, discretized with $N_x=2000$ and $N_y=500$ cells. The initial density is the indicator of the rectangle $[0.2,1.0]\times[0.6,1.3]$, evaluated at cell centres. Its discrete initial mass is $M_h^0=0.56$.

The deviator is represented by the segment

$$
Q=(0.6,2.0),\qquad P=(1.9,0.7),
$$

contained in the line $x+y=2.6$.

| Parameter | Value |
| --- | --- |
| Domain | $[0,4]\times[0,2]$ |
| Grid | $N_x=2000$, $N_y=500$ |
| Base belt speed | $v_T=0.8$ |
| Interaction coefficient | $\varepsilon=0.3$ |
| Gaussian parameter | $\sigma=10^4$ |
| Density threshold | $\rho_{\max}=1$ |
| Deviator endpoints | $Q=(0.6,2.0)$, $P=(1.9,0.7)$ |

## Construction of the transport field

The prescribed field is defined in `velocity_field.py`. It combines a smooth transition in the longitudinal direction with a Gaussian profile in the transverse direction. Near the deviator, a spatially weighted correction reduces the velocity component normal to the line $x+y=2.6$.

The geometric regions $A$, $B$ and $C$ are defined in the same module. The masks distinguish the part of the belt around the deviator and support the numerical analysis of transported mass in the selected regions.

Running `python velocity_field.py` visualizes the regions and the velocity vectors and reports values of $v_x+v_y$ along the segment, a numerical diagnostic of the normal component.

## Finite-volume simulation

The model is the [two-dimensional nonlocal conservation law](../../README.md#mathematical-model). The nonlocal field is obtained from convolutions of the density with derivatives of a Gaussian kernel and interpolated to cell interfaces. Each time step consists of an $x$ update followed by a $y$ update, with conservative Lax–Friedrichs-type fluxes, a common CFL-constrained time step and zero outer boundary fluxes.

The comparison involves two evolutions of the same initial density: the full nonlocal model and a reference transport model in which the nonlocal flux contribution is set to zero. The reference makes it possible to quantify the influence of the interaction for the chosen geometry.

## Running the experiment

Install the repository dependencies, then enter the deviator directory:

```bash
python -m pip install -r requirements.txt
cd experiments/deviator
```

For transport with the nonlocal contribution:

```bash
python main.py
```

This simulation runs to nominal time $T=1.5$ and writes `soluzioni_deviatore_tangente_con_nonlocale.npz`.

For reference transport:

```bash
python main_senza_nonlocale.py
```

This driver runs to nominal time $T=2$ and writes `soluzioni_deviatore_tangente.npz`. To compare the solutions at common recorded time steps:

```bash
python confronto_deviatore.py
```

The script reports discrete mass, density extrema, the $L^\infty$ norms of each solution and the $L^1$ and $L^\infty$ norms of their difference.

A checkpoint-enabled reference simulation is also available:

```bash
python main_senza_nonlocale_con_checkpoint.py
```

It uses the $2000\times500$ mesh, a nominal final time $T=1.5$, and saves `soluzioni_deviatore_tangente_senza_nonlocale_2000x500.npz`, periodic checkpoints and `tempi_deviatore_senza_nonlocale_2000x500.json`.

The observation times are approximately $t=0.01,0.1,0.5,1.0,1.5$. The exact numerical times are determined by the saved step indices and the time step $\Delta t$.

## Results and diagnostics

The numerical measurements from five saved states of the nonlocal experiment are reported in [`metriche_cinque_istanti_2000x500.csv`](metriche_cinque_istanti_2000x500.csv). They include the discrete mass

$$
M_h^n=\Delta x\,\Delta y\sum_{j,i}\rho_{j,i}^n,
$$

the density extrema, discrete variations in both spatial directions, and the mass contained in the disk of radius $0.20$ centred at $P$.

The tabulated mass is $0.56$ at each observation time up to floating-point precision. The recorded maxima provide additional information about local concentrations, while total variation and mass near the endpoint describe changes in the spatial distribution.

| Script | Analysis |
| --- | --- |
| `confronto_deviatore.py` | $L^1$ and $L^\infty$ differences between the nonlocal and reference solutions |
| `check_deviatore.py` | Mass, density extrema and location of the maximum in reference transport |
| `check_regione_B.py` | Mass and percentage of material in region $B$ |
| `check_flusso_deviatore.py` | Numerical flux across a cut near the deviator |
| `test_step.py` | A single splitting step, mass checks and positivity |
| `analisi_metriche.py` | Post-processing of saved snapshots and available checkpoint states |
| `plot_risultati_2000x500.py`, `plot_risultati_2000x500_senza.py` | Density maps showing the deviator segment |

For plotting the full-resolution solutions, the corresponding scripts expect the filenames `soluzioni_deviatore_tangente_con_nonlocale_2000x500.npz` and `soluzioni_deviatore_tangente_senza_nonlocale_2000x500.npz`. The direct nonlocal driver writes a shorter filename; a copy with the `_2000x500.npz` suffix can be used for the plotting and metrics scripts.

Large simulation arrays and checkpoint files are generated locally. The repository includes the [metrics table](metriche_cinque_istanti_2000x500.csv) and [reference-run timing metadata](tempi_deviatore_senza_nonlocale_2000x500.json) alongside the numerical code.
