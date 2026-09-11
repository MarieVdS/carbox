# Tutorial: running a CSE outflow model with Carbox

This walks through the two scripts in `cse/`:

| Script | What it does |
| --- | --- |
| `run_cse.py` | Integrate one circumstellar-envelope (CSE) outflow and write abundances / derivatives / rates. |
| `run_sensitivity.py` | Same outflow setup, but instead of writing a time series it differentiates the whole solve and reports a per-species **uncertainty budget** (rate coefficients + parent abundances). |
| `compare_shielding.py` | Helper: run the same outflow three ways (dust-only, vdb, oneband) and plot CO vs radius. |

Both main scripts share the **same physical setup** (`build_cse_config` in `run_cse.py`), so every `--network` / outflow / radiation / shielding / solver flag below means the same thing in both.

---

## 0. Setup

Use the project-local conda env for everything:

```bash
cd /Users/marie/Chemistry/carbox
.conda/bin/pip install -e .[dev]      # once
```

The scripts import each other (`run_sensitivity.py` does `from run_cse import ...`), so **run them from inside `cse/`**:

```bash
cd cse
../.conda/bin/python run_cse.py --network umist
```

Output lands in `cse/results/` by default (relative `--output` paths resolve against the script directory, not your shell's cwd).

---

## 1. What a CSE run actually is

The physics comes from `CSEPhysics` (`carbox/physics.py`): a constant-velocity spherical wind.

- **Radius vs time:** `r(t) = r_init + vexp * t`
- **Density:** mass conservation → `n(r) = Mdot / (4 pi r^2 vexp mu mH)`, i.e. `n ~ r^-2`
- **Temperature:** power law `T(r) = t_star * (r / r_star)^-eps`
- **Extinction:** from the radial column, `Av = n r / 1.87e21`
- **ODE state** is *fractional* abundance `x_i = n_i / n(t)`, so the expansion-dilution term cancels analytically — you do not set a dilution term.

`t_end` is **not** a direct input. It is computed from how long the wind takes to expand from `r_init` to `r_final` at `vexp`:

```
t_end [yr] = (r_final - r_init) / vexp / seconds_per_year      # vexp in cm/s
```

So you control the integration span with `--r-init` / `--r-final` / `--vexp`, not with a time.

Snapshots are placed on a grid **log-spaced in radius** (via `CSEPhysics.time_grid`), not in time, to avoid clustering everything near `t=0`.

---

## 2. A normal CSE run

### Simplest possible

```bash
cd cse
../.conda/bin/python run_cse.py --network umist
```

This uses the full UMIST22 network with the default O-rich AGB wind
(`Mdot = 1e-5 Msun/yr`, `vexp = 1.5e6 cm/s` = 15 km/s, `r_init = 1e14 cm`,
`r_final = 1.1e17 cm`, `t_star = 2000 K`, `eps = 0.7`), CO self-shielding via
the Morris & Jura one-band approximation, 100 snapshots.

It prints a header (density / temperature / Av at `r_init`, the derived
`t_end`, snapshot count), compiles the network, integrates, and writes:

```
cse/results/cse_umist_abundances.csv     # time, radius, n, T, Av + every species (fractional abundance)
cse/results/cse_umist_derivatives.csv    # dy/dt at each snapshot
cse/results/cse_umist_rates.csv          # per-reaction rate at each snapshot, keyed by reaction_id
cse/results/cse_umist_metadata.json      # config + solver stats
cse/results/cse_umist_summary.txt        # human-readable summary
```

### A more customised run

```bash
../.conda/bin/python run_cse.py \
  --network umist \
  --run-name orich_slow_wind \
  --mdot 5e-6 --vexp 1.0e6 --t-star 2500 --eps 0.65 \
  --r-init 1e14 --r-final 5e17 \
  --cr-rate 1.0 --fuv-field 1.0 \
  --co-shielding oneband \
  --n-snapshots 300 --rtol 1e-6 \
  --no-rates
```

### Faster iteration

Use the mini network while you are setting things up — it solves in a second or two:

```bash
../.conda/bin/python run_cse.py --network umist_mini --n-snapshots 100
```

---

## 3. All flags for `run_cse.py`

Run `../.conda/bin/python run_cse.py --help` for the canonical list. Explanations below.

### Top-level

| Flag | Default | Meaning |
| --- | --- | --- |
| `--network {umist,umist_mini,uclchem}` | `umist` | Which reaction network + matching O-rich initial-abundance file. `umist` = full UMIST22; `umist_mini` = small subset (fast); `uclchem` = UCLCHEM gas-phase-only network. |
| `--output DIR` | `results` | Output directory. Relative paths resolve against `cse/`, not your cwd. |
| `--run-name NAME` | `cse_<network>` | Prefix for every output file. |

### CSE outflow physics — these define `CSEPhysics` and the integration span

| Flag | Default | Units | Meaning |
| --- | --- | --- | --- |
| `--mdot` | `1e-5` | Msun/yr | Mass-loss rate. Sets the density normalisation (`n ~ Mdot / r^2 vexp`). |
| `--vexp` | `1.5e6` | cm/s | Expansion velocity (= 15 km/s). Enters density, the `r(t)` mapping, **and** `t_end`, and is required by the one-band CO shielding. |
| `--t-star` | `2000.0` | K | Temperature at `r_star` — the normalisation of the `T ~ r^-eps` profile. |
| `--r-init` | `1e14` | cm | Inner radius: where the integration starts. |
| `--r-final` | `1.1e17` | cm | Outer radius: sets `t_end` (see §1). Not a `CSEPhysics` field. |
| `--r-star` | `5e13` | cm | Stellar radius — the reference radius for the temperature power law. |
| `--eps` | `0.7` | — | Temperature power-law exponent (`T ~ r^-eps`). |

### Radiation environment

| Flag | Default | Meaning |
| --- | --- | --- |
| `--cr-rate` | `1.0` | Cosmic-ray ionisation rate **relative to the standard ISM rate** (so `1.0` = standard, `10` = 10×). Maps to `SimulationConfig.cr_rate`. |
| `--fuv-field` | `1.0` | External FUV field in Draine units. The interstellar radiation field driving photoreactions (attenuated by the radial `Av`). |

### Self-shielding (CO / C / H2 photo reactions)

By default `run_simulation` rewrites the dust-only CO photodissociation rate
into a self-shielded rate term (radial column `N_i = n_i * r`).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--no-self-shielding` | (shielding **on**) | Disable all self-shielding; use plain dust-attenuated photo rates. |
| `--co-shielding {oneband,vdb}` | `oneband` | CO treatment. `oneband` = Morris & Jura (1983) one-band analytic approximation for a constant-velocity outflow (the classic circumstellar treatment; needs `vexp`). `vdb` = van Dishoeck & Black (1988) lookup table. |
| `--shield-c-ionization` | off | Also rewrite `C + PHOTON -> C+ + e-` to UCLCHEM's shielded rate. Off by default — UCLCHEM's prescription is not right for every CSE. |
| `--shield-h2` | off | Also rewrite `H2 + PHOTON -> H + H` to a self-shielded rate. Off by default. |

Note: `--co-shielding oneband` **requires** a CSE physics model (it needs an
outflow velocity). It will raise if used on a static cloud. `vdb` works for both.

### Solver

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-snapshots` | `100` | Number of output radii/times (log-spaced in radius). |
| `--rtol` | `1e-5` | Relative tolerance. CSE runs tolerate looser rtol than static clouds because the state is fractional abundance. |
| `--atol` | `1e-20` | Absolute tolerance (floor on tracked abundances). |
| `--solver-name` | `kvaerno5` | Diffrax solver: `kvaerno5` / `kvaerno3` (implicit, stiff — recommended), `dopri5` / `tsit5` (explicit). |
| `--linear-solver {sparse,lu}` | `sparse` | Linear solve inside the implicit steps. `sparse` scales better for big networks; `lu` can be more robust for small ones. |
| `--max-steps` | `65536` | Cap on internal solver steps. Raise if the solver reports "max steps reached". |

### Output

| Flag | Meaning |
| --- | --- |
| `--no-derivatives` | Skip writing `*_derivatives.csv`. |
| `--no-rates` | Skip writing `*_rates.csv` (the big one — one column per reaction per snapshot). |

`*_abundances.csv`, `*_metadata.json`, `*_summary.txt` are always written.

---

## 4. Sensitivity / uncertainty analysis

`run_sensitivity.py` builds the **exact same outflow** as `run_cse.py`, then
instead of a time series it computes an **uncertainty budget** for the
predicted abundances by differentiating the whole ODE solve
(`jax.jacrev` through `solve_network`).

Three independent uncertainty sources:

1. **Rate coefficients** — `d(abundance)/d ln k_j` × `ln(uncertainty_factor_j)`.
   For UMIST, `uncertainty_factor` comes from the parsed A–E accuracy class of
   each reaction.
2. **Parent abundances** — `d(abundance)/d ln x_k(0)` × `ln(factor_k)`.
   The per-parent factor comes from the `uncertainties:` block of the
   initial-conditions YAML (see §6), or `--parent-uncertainty` for parents not
   listed there.
3. **Physical parameters** — `d(abundance)/d ln θ` × `ln(factor_θ)` for the CSE
   outflow parameters `mdot`, `vexp`, `t_star`, `eps`. **Opt-in:** contributes
   only when at least one of `--mdot-uncertainty` / `--vexp-uncertainty` /
   `--tstar-uncertainty` / `--eps-uncertainty` / `--physics-uncertainty` is set
   (and `--source` includes `physics`, which the default `all` does).
   `vexp` is perturbed at **fixed outer radius** — `t_end` is recomputed so the
   outflow still ends at the same `r_final`, and (with `t_start = 0`) the
   perturbed run lands on the same radius grid, so the derivative is
   `∂x(r)/∂ ln vexp` at fixed radius.

Per `(species, radius)` the individual shifts are quadrature-summed within
each source, then the sources are quadrature-summed into `sigma_total` and
`relative_uncertainty = sigma_total / nominal_abundance`.

> **Linearisation caveat.** This is a *first-order* (local) derivative.
> It is faithful for percent-level input spreads; for a factor-of-several
> excursion (e.g. an order-of-magnitude `mdot`, as in
> [Van de Sande et al. 2023](https://arxiv.org/abs/2304.05924)) the true
> response is non-linear and a model grid is the honest tool. `sigma_from_physics`
> then tells you *which* parameter dominates and the local slope, not an exact
> band.

**Cost:** reverse-mode AD gives the gradient w.r.t. *every* reaction, *every*
parent **and** every physical parameter in one backward pass — all three
sources cost the same as one. The real cost lever is the number of output
components = `len(--species) × n_snapshots`. Differentiating **all** species is
`O(n_species)` passes — slow. Always pass `--species` unless you really want the
whole network.

### Basic run

```bash
cd cse
../.conda/bin/python run_sensitivity.py --network umist --species H2O SiO HCN CO \
    --mdot-uncertainty 3 --vexp-uncertainty 1.3 --eps-uncertainty 1.2
```

Writes to `cse/results/`:

```
cse_umist_sensitivity_summary.csv   # one row per (species, snapshot): nominal, sigma_from_rates,
                                    #   sigma_from_parents, sigma_from_physics, sigma_total,
                                    #   worst_case_*, n_*, relative_uncertainty
cse_umist_sensitivity_rates.csv     # detail: contribution of each reaction to each species' error bar
cse_umist_sensitivity_parents.csv   # detail: contribution of each parent to each species' error bar
cse_umist_sensitivity_physics.csv   # detail: d(abundance)/d ln θ and shift for mdot/vexp/t_star/eps
```

and prints the ranked top contributors + the summary table.

### vs radius (one solve, every snapshot)

```bash
../.conda/bin/python run_sensitivity.py --network umist --species H2O SiO --snapshot-index all \
    --physics-uncertainty 2
```

`sigma_from_physics` vs radius is the interesting output here — the physical
parameters mostly matter through the density profile, so their weight grows
outward.

### One source only

```bash
../.conda/bin/python run_sensitivity.py --network umist --species H2O --source parents
../.conda/bin/python run_sensitivity.py --network umist --species H2O --source rates
../.conda/bin/python run_sensitivity.py --network umist --species H2O CO --source physics \
    --physics-uncertainty 2
```

### Inspect a single contributor

The full per-contributor detail tables (`*_sensitivity_{rates,parents,physics}.csv`)
are **always written in full** — one solve gives you the derivative w.r.t.
every reaction, parent and parameter, so there is nothing to gain by discarding
rows before saving. To look at one contributor, filter the CSV afterwards, e.g.:

```python
import pandas as pd
rates = pd.read_csv("cse/results/cse_umist_sensitivity_rates.csv")
rates[rates.reaction_id == 8259]                     # one reaction
rates[rates.species == "SiO"].nlargest(10, "uncertainty_shift", keep="all")

phys = pd.read_csv("cse/results/cse_umist_sensitivity_physics.csv")
phys[phys.parameter == "mdot"]                       # mdot's effect on every species
```

The terminal print-out already shows the top `--top-n` contributors ranked by
`|uncertainty_shift|`.

---

## 5. All flags for `run_sensitivity.py`

Everything from §3 **except** `--no-derivatives` / `--no-rates` (this script
never writes a time series — it forces `save_derivatives=False,
save_rates=False`), **plus** the group below.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--species [S ...]` | all species | Species to report. **Cost scales with this count** — always set it. Space-separated, e.g. `--species H2O SiO HCN`. |
| `--source {rates,parents,physics,both,all}` | `all` | Which uncertainty source(s) to propagate. `all` = rates + parents + physics; `both` = rates + parents (legacy). All sources are free relative to one. |
| `--snapshot-index N` | `-1` | Which snapshot to evaluate at. `-1` = last (outermost radius). Any integer indexes the log-radius grid. `all` = every snapshot (uncertainty vs radius from one solve). |
| `--parent-uncertainty F` | `1.5` | Default multiplicative factor for parents **without** an entry in the IC file's `uncertainties:` block ("believed within `[x/F, x*F]`"). The IC file's `default:` key, if present, overrides this. |
| `--mdot-uncertainty F` | *(unset)* | Multiplicative factor on `mdot` (`[mdot/F, mdot·F]`). Unset ⇒ falls back to `--physics-uncertainty`. |
| `--vexp-uncertainty F` | *(unset)* | Factor on `vexp`. Perturbed at fixed outer radius (`t_end` recomputed). |
| `--tstar-uncertainty F` | *(unset)* | Factor on `t_star` (temperature at `r_star`). |
| `--eps-uncertainty F` | *(unset)* | Factor on `eps`, the `T ∝ r^-eps` exponent. `F = 1.15` ≈ `eps ∈ [0.61, 0.81]` for `eps₀ = 0.7`. |
| `--physics-uncertainty F` | `1.0` | Shared default factor for any physics parameter not set individually. `1.0` = no uncertainty. Set this alone to give all four the same spread. |
| `--min-shift X` | `0.0` | Drop detail-table rows with `abs(uncertainty_shift) < X` before writing the CSV. A size cap for huge networks only — leave at `0` to keep the complete table. Summary unaffected. |
| `--top-n N` | `20` | How many ranked detail rows to print to the terminal (does not affect the CSVs). |

Same top-level `--output` / `--run-name` / `--network` as `run_cse.py`. The physics
factors can also be set in code / a config YAML via
`SimulationConfig.physics_uncertainties` (a `{name: factor}` dict with an optional
`"default"` key), the same way `parent_uncertainties` mirrors the IC-file block.

---

## 6. Initial conditions & the `uncertainties:` block

Each `--network` maps to a YAML in `benchmarks/initial_conditions/`
(`orich_cse_umist.yaml`, `orich_cse_umist_mini.yaml`,
`orich_cse_uclchem.yaml`). Structure:

```yaml
abundances:            # fractional, relative to H-nuclei density; becomes config.initial_abundances
  H2:   0.5
  CO:   1.50E-04
  H2O:  1.075e-04
  SiO:  1.355e-05
  ...

uncertainties:         # optional; only read by run_sensitivity.py / carbox.sensitivity
  default: 2.0         # factor for any parent not named below (overrides --parent-uncertainty)
  H:    1.0            # factor 1.0 == "no uncertainty", excludes this parent from the budget
  H2:   1.0
  He:   1.0
  e-:   1.0
  H2O:  2.0            # H2O believed within [x/2, x*2]
```

To change the outflow's chemistry (different C/O ratio, extra parents, etc.)
edit the `abundances:` block. To change the assumed observational error bars,
edit `uncertainties:`. Delete the `uncertainties:` block entirely to fall back
to `--parent-uncertainty` for every parent.

---

## 7. Comparing CO self-shielding treatments

```bash
cd cse
../.conda/bin/python compare_shielding.py --network umist --species CO
```

Runs the outflow three ways — `dust_only` (`--no-self-shielding`), `vdb`,
`oneband` — and writes `*_shielding_comparison.csv` + `.png` of CO vs radius.
Takes the same `--network` / physics / solver flags.

---

## 8. Driving it from Python

Both scripts are thin wrappers. For notebooks / parameter sweeps:

```python
from run_cse import build_cse_config, run_cse

# just build the config (no solve) — e.g. to reuse a compiled network across a sweep
input_file, fmt, run_name, out_dir, config = build_cse_config(
    network="umist", mdot=5e-6, vexp=1.0e6, r_final=5e17, n_snapshots=200,
)

# or build + solve + write output in one call
results = run_cse(network="umist", species=None, mdot=5e-6, run_name="my_run")
solution = results["solution"]      # diffrax solution: solution.ts, solution.ys
network  = results["network"]
```

For the uncertainty budget directly:

```python
from carbox.sensitivity import uncertainty_budget
budget = uncertainty_budget(network, jnetwork, y0, config,
                            species=["H2O", "SiO"], snapshot_index="all",
                            sources=("rates", "parents", "physics"),
                            physics_uncertainty={"mdot": 3.0, "vexp": 1.3,
                                                 "t_star": 1.2, "eps": 1.15})
budget.summary      # DataFrame
budget.rates        # per-reaction detail
budget.parents      # per-parent detail
budget.physics      # per-parameter detail (mdot / vexp / t_star / eps)
```

`carbox.sensitivity.physical_parameter_sensitivity(...)` returns the physics
detail table on its own, mirroring `rate_coefficient_sensitivity` /
`initial_abundance_sensitivity`.

See `carbox/main.py` (`parse_network` / `solve`) for reusing one compiled
network across many solves with different `rate_modifiers`.
