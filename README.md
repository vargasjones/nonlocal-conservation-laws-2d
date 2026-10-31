# Two-dimensional nonlocal conservation laws: finite-volume simulations

*Python implementation developed for my master's thesis in Mathematics at the University of Milano-Bicocca.*

This work examines the transport of material on a conveyor belt through a two-dimensional conservation law. The motion is governed by an assigned velocity field and a nonlocal contribution determined by the surrounding material density. The numerical experiments investigate the influence of this interaction in two configurations: straight transport and transport in the presence of a deviator.

The repository contains the finite-volume Lax–Friedrichs implementation, simulation drivers for both configurations, and numerical diagnostics.

## Mathematical model

The material density $\rho=\rho(t,x,y)$ satisfies

$$
\partial_t\rho+
\nabla\cdot\left(\rho\,v^{\mathrm{stat}}+
f(\rho)\,\mathcal I(\rho)\right)=0,
$$

where $v^{\mathrm{stat}}$ is the prescribed transport velocity and

$$
\mathcal I(\rho)=
-\varepsilon\frac{\nabla(\eta*\rho)}
{\sqrt{1+|\nabla(\eta*\rho)|^2}}.
$$

The convolution kernel is Gaussian,

$$
\eta(z)=\frac{\sigma}{2\pi}
\exp\left(-\frac{\sigma}{2}|z|^2\right),
\qquad z\in\mathbb R^2,
$$

and the density response used in the implementation is

$$
f(\rho)=\rho\left[
\frac12+\frac1\pi\arctan\bigl(50(\rho-\rho_{\max})\bigr)
\right].
$$

The parameter $\varepsilon$ controls the nonlocal contribution. The term $\nabla(\eta*\rho)$ incorporates the spatial distribution of material into the flux.

## Numerical method

I discretize the conservation law on a Cartesian mesh using a **Lax–Friedrichs-type finite-volume scheme** with an $x$-then-$y$ dimensional splitting. At each full time step, the components of $\mathcal I(\rho^n)$ are computed by discrete convolutions with the Gaussian derivatives, then averaged from cell centres to the corresponding interfaces. The density is subsequently updated in the two directions.

For adjacent cell values $u,w$, the $x$-direction flux is

$$
F(u,w;v,J)
=\frac12\left[v(u+w)+J\bigl(f(u)+f(w)\bigr)\right]
-\frac{\alpha}{2}(w-u).
$$

The $y$-direction flux has the analogous form, with numerical viscosity coefficient $\beta$. The coefficients depend on the prescribed field, $\varepsilon$ and a numerical estimate of the Lipschitz constant of $f$. The same time step, selected from restrictions in both directions, is used for the two updates.

The implementation prescribes zero numerical flux at the outer boundary. The flux-difference form therefore conserves discrete total mass up to floating-point error.

The nonlocal term is computed with `scipy.signal.oaconvolve`. Density arrays have shape `(Ny, Nx)`: the first index is associated with $y$, the second with $x$.

## Experiments

### Straight-belt transport

The first configuration studies material moving under the uniform prescribed velocity $v^{\mathrm{stat}}=(0.42,0)$. Starting from a rectangular region of density one, I compare the evolution with the nonlocal term against a reference transport evolution.

| Quantity | Value |
| --- | --- |
| Domain | $[0,10]\times[0,5]$ |
| Grid | $N_x=2000$, $N_y=500$ |
| Initial density | $\mathbf1_{[2,8]\times[1,4]}$ at cell centres |
| Nonlocal coefficient | $\varepsilon=0.3$ |
| Gaussian parameter | $\sigma=10^4$ |
| Density threshold | $\rho_{\max}=1$ |
| Nominal final time | $T=1$ |

The drivers `main.py` and `main2.py` implement the nonlocal and reference cases, respectively. They save the density at time steps close to $t=0.01,0.1,0.5,1.0$.

### Transport with a deviator

The second configuration uses a spatially varying, regularized prescribed velocity on $[0,4]\times[0,2]$. The deviator is the segment joining $Q=(0.6,2.0)$ to $P=(1.9,0.7)$. The numerical investigation concerns the redistribution of material near the segment and the effect of nonlocal interactions on the density profile.

The code, geometric construction, diagnostics and reproduction instructions for this case are in **[experiments/deviator](experiments/deviator/README.md)**.

## Reproducing the straight-belt experiment

The project uses Python 3.13, NumPy, SciPy and Matplotlib. From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run both transport models:

```bash
python main.py
python main2.py
```

They produce `soluzioni.npz` and `soluzioni_senza_nonlocale.npz`. The following scripts analyse their saved snapshots:

```bash
python plot_results.py
python check_massa_senzanonlocal.py
python plot_difference.py
```

The post-processing includes density plots, the discrete mass

$$
M_h^n=\Delta x\,\Delta y\sum_{j,i}\rho_{j,i}^n,
$$

the discrete maximum norm and pointwise comparisons of the two solutions. Both simulations must use the **same spatial grid, initial data and time-step parameters** for a meaningful difference calculation.

The full-resolution computation is demanding, especially because the nonlocal field requires repeated two-dimensional convolutions. The parameters in `parameters.py` can be adjusted for exploratory runs. Output arrays are generated locally rather than stored in the repository.

For the second experiment, see the [deviator execution instructions](experiments/deviator/README.md#running-the-experiment).

## Code

| Files | Purpose |
| --- | --- |
| `parameters.py`, `grid.py`, `initial_data.py` | Spatial mesh, domain and initial density |
| `velocity_field.py`, `heaviside.py` | Prescribed field and density response |
| `kernel.py`, `nonlocal_term.py` | Gaussian kernel and nonlocal convolutions |
| `lax_friedrichs.py`, `cfl.py` | Numerical fluxes, splitting steps and time-step restrictions |
| `main.py`, `main2.py` | Straight-belt simulations |
| `plot_results.py`, `plot_difference.py`, `check_massa_senzanonlocal.py` | Diagnostics and comparison of saved solutions |
| `test_x_step.py`, `test_full_step.py`, `benchmark_conv.py` | Numerical tests and convolution timing |
| [`experiments/deviator/`](experiments/deviator/README.md) | Deviator experiment and numerical results |

## Thesis context

*Buona posizione e approssimazione ai volumi finiti per leggi di conservazione non locali su domini bidimensionali* — Master's thesis in Mathematics, University of Milano-Bicocca.

The thesis addresses the well-posedness and finite-volume approximation of nonlocal conservation laws. This repository contains the Python implementation used for the numerical study.
