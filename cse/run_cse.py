#!/usr/bin/env python3
"""
Run Carbox for a circumstellar-envelope (CSE) outflow.

Wraps ``carbox`` with a ``CSEPhysics`` model (constant-velocity spherical
expansion): density follows ``n ~ r^-2``, temperature ``T ~ r^-eps``, Av from
the radial column, and ``r(t) = r_init + vexp * t``. The ODE state is
fractional abundances, so expansion dilution cancels analytically.

``build_cse_config`` returns the resolved network file plus a ready
``SimulationConfig`` without solving anything -- ``run_cse`` solves and
writes output, and ``run_sensitivity.py`` reuses ``build_cse_config`` so the
sensitivity analysis runs on an identical outflow setup.

Examples
--------
    python run_cse.py --network umist
    python run_cse.py --network umist_mini --mdot 5e-6 --vexp 1.0e6
    python run_cse.py --network umist --r-final 5e17 --n-snapshots 200
"""

import argparse
from pathlib import Path

import jax
import yaml

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.physics import CSEPhysics
from carbox.solver import SPY

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Network choices. Each maps to a reaction-network file, its format, and the
# O-rich parent-species initial abundances (fractional, relative to H nuclei).
CSE_NETWORKS = {
    "umist": {
        "description": "Full UMIST22 network, O-rich CSE",
        "input_file": REPO_ROOT / "data" / "umist22.csv",
        "input_format": "umist",
        "initial_conditions": REPO_ROOT / "benchmarks" / "initial_conditions" / "orich_cse_umist.yaml",
    },
    "umist_mini": {
        "description": "Small UMIST22 subset, O-rich CSE",
        "input_file": REPO_ROOT / "data" / "umist22_mini.csv",
        "input_format": "umist",
        "initial_conditions": REPO_ROOT / "benchmarks" / "initial_conditions" / "orich_cse_umist_mini.yaml",
    },
    "uclchem": {
        "description": "UCLCHEM gas-phase-only network, O-rich CSE",
        "input_file": REPO_ROOT / "data" / "uclchem_gas_phase_only.csv",
        "input_format": "uclchem",
        "initial_conditions": REPO_ROOT / "benchmarks" / "initial_conditions" / "orich_cse_uclchem.yaml",
    },
}

# Default CSE outflow parameters (O-rich AGB wind).
DEFAULT_PHYSICS = {
    "mdot": 1.0e-5,    # Mass-loss rate [Msun/yr]
    "vexp": 1.5e6,     # Expansion velocity [cm/s] (= 15 km/s)
    "t_star": 2000.0,  # Temperature at r_star [K]
    "r_init": 1.0e14,  # Initial radius [cm]
    "r_final": 1.1e17,  # Final radius [cm] (sets t_end, not a CSEPhysics field)
    "r_star": 5.0e13,  # Stellar radius, T-profile normalization [cm]
    "eps": 0.7,        # Temperature power-law exponent
}

DEFAULT_SOLVER = {
    "cr_rate": 1.0,       # relative to the standard ISM rate
    "fuv_field": 1.0,     # Draine units
    "n_snapshots": 100,
    "rtol": 1.0e-5,
    "atol": 1.0e-20,
    "solver_name": "kvaerno5",
    "linear_solver": "sparse",
    "max_steps": 65536,
}


def resolve_t_end_years(r_init: float, r_final: float, vexp: float) -> float:
    """Years to expand from ``r_init`` to ``r_final`` at constant ``vexp`` [cm/s]."""
    return (r_final - r_init) / vexp / SPY


def build_cse_config(
    network: str = "umist",
    output: str = "results",
    run_name: str = None,
    *,
    mdot: float = DEFAULT_PHYSICS["mdot"],
    vexp: float = DEFAULT_PHYSICS["vexp"],
    t_star: float = DEFAULT_PHYSICS["t_star"],
    r_init: float = DEFAULT_PHYSICS["r_init"],
    r_final: float = DEFAULT_PHYSICS["r_final"],
    r_star: float = DEFAULT_PHYSICS["r_star"],
    eps: float = DEFAULT_PHYSICS["eps"],
    cr_rate: float = DEFAULT_SOLVER["cr_rate"],
    fuv_field: float = DEFAULT_SOLVER["fuv_field"],
    n_snapshots: int = DEFAULT_SOLVER["n_snapshots"],
    rtol: float = DEFAULT_SOLVER["rtol"],
    atol: float = DEFAULT_SOLVER["atol"],
    solver_name: str = DEFAULT_SOLVER["solver_name"],
    linear_solver: str = DEFAULT_SOLVER["linear_solver"],
    max_steps: int = DEFAULT_SOLVER["max_steps"],
    self_shielding: bool = True,
    co_shielding_method: str = "oneband",
    shield_c_ionization: bool = False,
    shield_h2: bool = False,
    physics_uncertainties: dict = None,
    save_derivatives: bool = True,
    save_rates: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Resolve the network files and build a ``SimulationConfig`` for a CSE run
    without solving. Returns ``(input_file, format_type, run_name,
    output_dir, config)``.

    ``output`` resolves against this script's directory unless absolute, so
    results land in a predictable place regardless of the caller's cwd.
    """
    if network not in CSE_NETWORKS:
        raise ValueError(
            f"Unknown network '{network}'. Available: {list(CSE_NETWORKS)}"
        )
    net_cfg = CSE_NETWORKS[network]
    run_name = run_name or f"cse_{network}"

    input_file = net_cfg["input_file"]
    ic_file = net_cfg["initial_conditions"]
    if not input_file.exists():
        raise FileNotFoundError(f"Network file not found: {input_file}")
    if not ic_file.exists():
        raise FileNotFoundError(f"Initial conditions file not found: {ic_file}")

    with open(ic_file) as f:
        ic_data = yaml.safe_load(f)
    initial_abundances = ic_data["abundances"]
    # Optional `uncertainties:` block -> per-parent factor for
    # parent-abundance error propagation (carbox.sensitivity).
    parent_uncertainties = ic_data.get("uncertainties")

    physics = CSEPhysics(
        mdot=mdot, vexp=vexp, t_star=t_star, r_init=r_init, r_star=r_star, eps=eps
    )
    t_end_yr = resolve_t_end_years(r_init, r_final, vexp)

    output_dir = Path(output)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        n0, t0, av0, _ = physics.get_conditions(t_sec=0.0)
        print("=" * 70)
        print(f"Carbox CSE run: {network} ({run_name})")
        print("=" * 70)
        print(f"Network: {input_file}")
        print(
            f"Outflow: mdot={mdot:.2e} Msun/yr, vexp={vexp:.3e} cm/s "
            f"({vexp / 1.0e5:.1f} km/s), r_init={r_init:.2e} -> r_final={r_final:.2e} cm"
        )
        print(
            f"At r_init: n={float(n0):.3e} cm^-3, T={float(t0):.1f} K, Av={float(av0):.2f} mag"
        )
        print(f"Integration: 0 -> {t_end_yr:.3e} yr, {n_snapshots} snapshots")
        print()

    config = SimulationConfig(
        cr_rate=cr_rate,
        fuv_field=fuv_field,
        t_start=0.0,
        t_end=t_end_yr,
        n_snapshots=n_snapshots,
        rtol=rtol,
        atol=atol,
        solver=solver_name,
        linear_solver=linear_solver,
        max_steps=max_steps,
        self_shielding=self_shielding,
        co_shielding_method=co_shielding_method,
        shield_c_ionization=shield_c_ionization,
        shield_h2=shield_h2,
        parent_uncertainties=parent_uncertainties,
        physics_uncertainties=physics_uncertainties,
        output_dir=str(output_dir),
        run_name=run_name,
        save_abundances=True,  # always includes physics columns (n, T, Av, r)
        save_derivatives=save_derivatives,
        save_rates=save_rates,
        initial_abundances=initial_abundances,
        physics_model=physics,
    )
    return input_file, net_cfg["input_format"], run_name, output_dir, config


def run_cse(**kwargs) -> dict:
    """Build a CSE config (see ``build_cse_config``) and solve it."""
    verbose = kwargs.get("verbose", True)
    input_file, format_type, run_name, output_dir, config = build_cse_config(**kwargs)

    results = run_simulation(
        network_file=str(input_file),
        config=config,
        format_type=format_type,
        verbose=verbose,
    )

    if verbose:
        print(f"\nOutputs in {output_dir}:")
        for suffix in ("abundances", "derivatives", "rates", "metadata", "summary"):
            print(f"  {run_name}_{suffix}.*")

    results["output_dir"] = output_dir
    results["run_name"] = run_name
    return results


def add_common_cse_args(parser: argparse.ArgumentParser) -> None:
    """Shared ``--network``/physics/solver args for run_cse.py and
    run_sensitivity.py, so both take the same physical setup."""
    parser.add_argument(
        "--network", default="umist", choices=list(CSE_NETWORKS), help="Reaction network"
    )

    phys = parser.add_argument_group("CSE outflow physics")
    phys.add_argument("--mdot", type=float, default=DEFAULT_PHYSICS["mdot"], help="Mass-loss rate [Msun/yr]")
    phys.add_argument("--vexp", type=float, default=DEFAULT_PHYSICS["vexp"], help="Expansion velocity [cm/s]")
    phys.add_argument("--t-star", type=float, default=DEFAULT_PHYSICS["t_star"], help="Temperature at r_star [K]")
    phys.add_argument("--r-init", type=float, default=DEFAULT_PHYSICS["r_init"], help="Initial radius [cm]")
    phys.add_argument("--r-final", type=float, default=DEFAULT_PHYSICS["r_final"], help="Final radius [cm] (sets t_end)")
    phys.add_argument("--r-star", type=float, default=DEFAULT_PHYSICS["r_star"], help="Stellar radius, T normalization [cm]")
    phys.add_argument("--eps", type=float, default=DEFAULT_PHYSICS["eps"], help="Temperature power-law exponent")

    env = parser.add_argument_group("Radiation environment")
    env.add_argument("--cr-rate", type=float, default=DEFAULT_SOLVER["cr_rate"], help="Cosmic-ray rate, relative to standard")
    env.add_argument("--fuv-field", type=float, default=DEFAULT_SOLVER["fuv_field"], help="FUV field [Draine units]")

    shield = parser.add_argument_group("Self-shielding (CO / C / H2 photo reactions)")
    shield.add_argument("--no-self-shielding", dest="self_shielding", action="store_false",
                        help="Disable self-shielding; use the dust-only photo rates")
    shield.add_argument("--co-shielding", dest="co_shielding_method", default="oneband",
                        choices=["vdb", "oneband"],
                        help="CO self-shielding: 'oneband' (Morris & Jura 1983, the "
                             "standard circumstellar treatment; default) or 'vdb' "
                             "(van Dishoeck & Black 1988 table)")
    shield.add_argument("--shield-c-ionization", dest="shield_c_ionization",
                        action="store_true",
                        help="Also rewrite C -> C+ + e- to UCLCHEM's shielded rate (off by default)")
    shield.add_argument("--shield-h2", dest="shield_h2", action="store_true",
                        help="Also rewrite H2 -> H + H to a self-shielded rate (off by default)")

    solver = parser.add_argument_group("Solver")
    solver.add_argument("--n-snapshots", type=int, default=DEFAULT_SOLVER["n_snapshots"])
    solver.add_argument("--rtol", type=float, default=DEFAULT_SOLVER["rtol"])
    solver.add_argument("--atol", type=float, default=DEFAULT_SOLVER["atol"])
    solver.add_argument("--solver-name", dest="solver_name", default=DEFAULT_SOLVER["solver_name"])
    solver.add_argument("--linear-solver", default=DEFAULT_SOLVER["linear_solver"], choices=["lu", "sparse"])
    solver.add_argument("--max-steps", type=int, default=DEFAULT_SOLVER["max_steps"])


def main():
    parser = argparse.ArgumentParser(
        description="Run Carbox for a CSE (circumstellar envelope) outflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Networks:\n"
        + "\n".join(f"  {n:<12} - {c['description']}" for n, c in CSE_NETWORKS.items()),
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory (relative paths resolve against this script's directory)",
    )
    parser.add_argument("--run-name", default=None, help="Overrides the default run name")
    add_common_cse_args(parser)

    out = parser.add_argument_group("Output")
    out.add_argument("--no-derivatives", action="store_true", help="Skip writing dy/dt CSV")
    out.add_argument("--no-rates", action="store_true", help="Skip writing per-reaction rates CSV")

    args = vars(parser.parse_args())
    args["save_derivatives"] = not args.pop("no_derivatives")
    args["save_rates"] = not args.pop("no_rates")
    run_cse(**args)


if __name__ == "__main__":
    main()
