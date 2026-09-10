#!/usr/bin/env python3
"""
Uncertainty budget for a CSE run's predicted abundances.

Differentiates the whole ODE solve (``jax.jacrev`` through ``solve_network``)
with respect to three independent uncertainty sources and combines each
derivative with the assumed input spread:

- **rate coefficients** -- ``d(abundance)/d ln k_j`` x ``ln(uncertainty_factor_j)``
  (for UMIST, the A-E accuracy class);
- **parent abundances** -- ``d(abundance)/d ln x_k(0)`` x ``ln(factor_k)``
  (from the ``uncertainties:`` block of the initial-conditions YAML, or
  ``--parent-uncertainty``);
- **physical parameters** -- ``d(abundance)/d ln theta`` x ``ln(factor_theta)``
  for the CSE outflow parameters ``mdot`` / ``vexp`` / ``t_star`` / ``eps``
  (from ``--mdot-uncertainty`` / ``--vexp-uncertainty`` / ``--tstar-uncertainty``
  / ``--eps-uncertainty``, or ``--physics-uncertainty`` as the shared default).
  ``vexp`` is perturbed at fixed outer radius (``t_end`` is recomputed).

Per (species, radius) the shifts are quadrature-summed within each source and
the sources are quadrature-summed into ``sigma_total`` /
``relative_uncertainty``. This is a first-order (local) linearisation --
faithful for percent-level spreads, only approximate for a factor-of-several
``mdot`` (where a model grid is the honest tool).

The outflow setup is exactly the one ``run_cse.py`` uses; the same
``--network`` / ``--mdot`` / ... flags apply.

Cost: reverse-mode AD gives the gradient w.r.t. every reaction, every parent
AND every physical parameter in one pass -- all three sources together cost
the same as one. The levers are ``--species`` (cost ~ number of species) and
``--snapshot-index`` (``all`` keeps every radius).

Examples
--------
    # All three sources, a few species, final radius
    python run_sensitivity.py --network umist --species H2O SiO HCN CO \
        --mdot-uncertainty 3 --vexp-uncertainty 1.3 --eps-uncertainty 1.2

    # vs radius, from one solve
    python run_sensitivity.py --network umist --species H2O SiO --snapshot-index all

    # Parent-abundance uncertainty only
    python run_sensitivity.py --network umist --species H2O --source parents

    # Physical parameters only, one shared factor
    python run_sensitivity.py --network umist --species H2O CO --source physics \
        --physics-uncertainty 2

The full per-contributor detail tables are always written to CSV
(``*_sensitivity_{rates,parents,physics}.csv``); filter them afterwards to
inspect one reaction / parent / parameter.
"""

import argparse

import pandas as pd

from carbox.initial_conditions import initialize_abundances
from carbox.parsers import parse_chemical_network
from carbox.sensitivity import (
    DEFAULT_PARENT_UNCERTAINTY,
    DEFAULT_PHYSICS_UNCERTAINTY,
    uncertainty_budget,
)
from carbox.shielding import configure_self_shielding

from run_cse import add_common_cse_args, build_cse_config


def _snapshot_index(value: str):
    return "all" if value == "all" else int(value)


def _print_ranked(df, label, top_n):
    if df is None or df.empty:
        return
    ranked = df.reindex(df["uncertainty_shift"].abs().sort_values(ascending=False).index)
    print(f"\nTop {min(top_n, len(ranked))} {label} by |uncertainty_shift|:")
    with pd.option_context("display.max_rows", top_n, "display.width", 180):
        print(ranked.head(top_n).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory (relative paths resolve against this script's directory)",
    )
    parser.add_argument("--run-name", default=None, help="Overrides the default run name")
    add_common_cse_args(parser)

    sens = parser.add_argument_group("Uncertainty budget")
    sens.add_argument(
        "--species", nargs="*", default=None,
        help="Species to report (default: all -- cost scales with this count)",
    )
    sens.add_argument(
        "--source", choices=["rates", "parents", "physics", "both", "all"],
        default="all",
        help="Uncertainty source(s) to propagate: 'all' (default) = rates + "
             "parents + physics; 'both' = rates + parents (legacy)",
    )
    sens.add_argument(
        "--snapshot-index", type=_snapshot_index, default=-1,
        help="Snapshot to evaluate at (default -1, the last), or 'all' for every radius",
    )
    sens.add_argument(
        "--parent-uncertainty", type=float, default=DEFAULT_PARENT_UNCERTAINTY,
        help="Default multiplicative factor for parents without an entry in the "
             f"IC file's uncertainties block (default {DEFAULT_PARENT_UNCERTAINTY})",
    )
    sens.add_argument(
        "--min-shift", type=float, default=0.0,
        help="Drop detail rows with abs(uncertainty_shift) below this (default 0)",
    )
    sens.add_argument("--top-n", type=int, default=20, help="Detail rows to print")

    phys = parser.add_argument_group("Physical-parameter uncertainty")
    phys.add_argument("--mdot-uncertainty", type=float, default=None,
                      help="Multiplicative factor F on mdot ([mdot/F, mdot*F])")
    phys.add_argument("--vexp-uncertainty", type=float, default=None,
                      help="Multiplicative factor F on vexp (fixed outer radius)")
    phys.add_argument("--tstar-uncertainty", type=float, default=None,
                      help="Multiplicative factor F on t_star")
    phys.add_argument("--eps-uncertainty", type=float, default=None,
                      help="Multiplicative factor F on the T power-law exponent eps")
    phys.add_argument(
        "--physics-uncertainty", type=float, default=DEFAULT_PHYSICS_UNCERTAINTY,
        help="Shared default factor for any physics parameter not set individually "
             f"(default {DEFAULT_PHYSICS_UNCERTAINTY} = no uncertainty)",
    )

    args = vars(parser.parse_args())
    species = args.pop("species")
    source = args.pop("source")
    snapshot_index = args.pop("snapshot_index")
    parent_default = args.pop("parent_uncertainty")
    min_shift = args.pop("min_shift")
    top_n = args.pop("top_n")

    physics_default = args.pop("physics_uncertainty")
    physics_uncertainties = {"default": physics_default}
    for flag, name in (
        ("mdot_uncertainty", "mdot"), ("vexp_uncertainty", "vexp"),
        ("tstar_uncertainty", "t_star"), ("eps_uncertainty", "eps"),
    ):
        value = args.pop(flag)
        if value is not None:
            physics_uncertainties[name] = value

    sources = {
        "both": ("rates", "parents"),
        "all": ("rates", "parents", "physics"),
    }.get(source, (source,))

    if "physics" in sources and set(physics_uncertainties) == {"default"} \
            and physics_default == 1.0:
        print(
            "Note: physics source requested but no --*-uncertainty / "
            "--physics-uncertainty given -> sigma_from_physics will be 0."
        )

    if species is None:
        print(
            "No --species given: differentiating every species in the network "
            "(O(n_species) reverse-mode passes). Pass --species to speed this up."
        )

    input_file, format_type, run_name, output_dir, config = build_cse_config(
        save_derivatives=False, save_rates=False,
        physics_uncertainties=physics_uncertainties, **args
    )

    network = parse_chemical_network(str(input_file), format_type)
    configure_self_shielding(network, config)
    if getattr(network, "_n_self_shielded", 0):
        print(
            f"Self-shielding: {network._n_self_shielded} photo reaction(s) rewritten "
            f"(CO method: {getattr(network, '_co_shielding_method', config.co_shielding_method)})"
        )
    y0 = initialize_abundances(network, config)
    jnetwork = network.get_ode()

    print(
        f"\nUncertainty budget [{', '.join(sources)}] for "
        f"{species or 'all species'} at snapshot {snapshot_index} ..."
    )

    budget = uncertainty_budget(
        network, jnetwork, y0, config,
        species=species,
        snapshot_index=snapshot_index,
        sources=sources,
        parent_uncertainty_default=parent_default,
        min_abs_shift=min_shift,
    )

    if budget.rates is not None:
        path = output_dir / f"{run_name}_sensitivity_rates.csv"
        budget.rates.to_csv(path, index=False)
        _print_ranked(budget.rates, "reactions", top_n)
        print(f"Saved rate detail ({len(budget.rates)} rows): {path}")

    if budget.parents is not None:
        path = output_dir / f"{run_name}_sensitivity_parents.csv"
        budget.parents.to_csv(path, index=False)
        _print_ranked(budget.parents, "parents", top_n)
        print(f"Saved parent detail ({len(budget.parents)} rows): {path}")

    if budget.physics is not None:
        path = output_dir / f"{run_name}_sensitivity_physics.csv"
        budget.physics.to_csv(path, index=False)
        _print_ranked(budget.physics, "physical parameters", top_n)
        print(f"Saved physics detail ({len(budget.physics)} rows): {path}")

    summary_path = output_dir / f"{run_name}_sensitivity_summary.csv"
    budget.summary.to_csv(summary_path, index=False)
    print(
        "\nPer-species error bar (quadrature over ALL reactions / ALL parents / "
        "ALL physics params; sigma_total = sqrt(rates^2 + parents^2 + physics^2)):"
    )
    with pd.option_context("display.max_rows", len(budget.summary), "display.width", 200):
        print(budget.summary.to_string(index=False))
    print(f"\nSaved summary ({len(budget.summary)} rows): {summary_path}")


if __name__ == "__main__":
    main()
