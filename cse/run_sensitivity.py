#!/usr/bin/env python3
"""
Uncertainty budget for a CSE run's predicted abundances.

Differentiates the whole ODE solve (``jax.jacrev`` through ``solve_network``)
with respect to two independent uncertainty sources and combines each
derivative with the assumed input spread:

- **rate coefficients** -- ``d(abundance)/d ln k_j`` x ``ln(uncertainty_factor_j)``
  (for UMIST, the A-E accuracy class);
- **parent abundances** -- ``d(abundance)/d ln x_k(0)`` x ``ln(factor_k)``
  (from the ``uncertainties:`` block of the initial-conditions YAML, or
  ``--parent-uncertainty``).

Per (species, radius) the shifts are quadrature-summed within each source
and the two sources are quadrature-summed into ``sigma_total`` /
``relative_uncertainty``.

The outflow setup is exactly the one ``run_cse.py`` uses; the same
``--network`` / ``--mdot`` / ... flags apply.

Cost: reverse-mode AD gives the gradient w.r.t. every reaction AND every
parent in one pass, so both sources together cost the same as one. The
levers are ``--species`` (cost ~ number of species) and ``--snapshot-index``
(``all`` keeps every radius).

Examples
--------
    # Both sources, a few species, final radius
    python run_sensitivity.py --network umist --species H2O SiO HCN CO

    # vs radius, from one solve
    python run_sensitivity.py --network umist --species H2O SiO --snapshot-index all

    # Parent-abundance uncertainty only
    python run_sensitivity.py --network umist --species H2O --source parents

    # Inspect one contributor (summary sigma stays the full-set value)
    python run_sensitivity.py --network umist --species SiO --show-parent H2O
    python run_sensitivity.py --network umist --species CO --show-reaction-id 8259
"""

import argparse

import pandas as pd

from carbox.initial_conditions import initialize_abundances
from carbox.parsers import parse_chemical_network
from carbox.sensitivity import DEFAULT_PARENT_UNCERTAINTY, uncertainty_budget
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
        "--source", choices=["rates", "parents", "both"], default="both",
        help="Uncertainty source(s) to propagate (default: both)",
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
        "--show-reaction-id", type=int, nargs="+", default=None,
        help="Filter the rates detail table to these reactions (summary sigma is unaffected)",
    )
    sens.add_argument(
        "--show-parent", nargs="+", default=None,
        help="Filter the parents detail table to these parents (summary sigma is unaffected)",
    )
    sens.add_argument(
        "--min-shift", type=float, default=0.0,
        help="Drop detail rows with abs(uncertainty_shift) below this (default 0)",
    )
    sens.add_argument("--top-n", type=int, default=20, help="Detail rows to print")

    args = vars(parser.parse_args())
    species = args.pop("species")
    source = args.pop("source")
    snapshot_index = args.pop("snapshot_index")
    parent_default = args.pop("parent_uncertainty")
    show_reaction_id = args.pop("show_reaction_id")
    show_parent = args.pop("show_parent")
    min_shift = args.pop("min_shift")
    top_n = args.pop("top_n")

    sources = ("rates", "parents") if source == "both" else (source,)

    if species is None:
        print(
            "No --species given: differentiating every species in the network "
            "(O(n_species) reverse-mode passes). Pass --species to speed this up."
        )

    input_file, format_type, run_name, output_dir, config = build_cse_config(
        save_derivatives=False, save_rates=False, **args
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
        reaction_ids=show_reaction_id,
        parents=show_parent,
        parent_uncertainty_default=parent_default,
        min_abs_shift=min_shift,
    )

    if budget.rates is not None:
        path = output_dir / f"{run_name}_sensitivity_rates.csv"
        budget.rates.to_csv(path, index=False)
        _print_ranked(budget.rates, "reactions" if show_reaction_id is None
                      else f"reactions (filtered to {show_reaction_id})", top_n)
        print(f"Saved rate detail ({len(budget.rates)} rows): {path}")

    if budget.parents is not None:
        path = output_dir / f"{run_name}_sensitivity_parents.csv"
        budget.parents.to_csv(path, index=False)
        _print_ranked(budget.parents, "parents" if show_parent is None
                      else f"parents (filtered to {show_parent})", top_n)
        print(f"Saved parent detail ({len(budget.parents)} rows): {path}")

    summary_path = output_dir / f"{run_name}_sensitivity_summary.csv"
    budget.summary.to_csv(summary_path, index=False)
    print(
        "\nPer-species error bar "
        "(quadrature over ALL reactions / ALL parents; sigma_total = "
        "sqrt(rates^2 + parents^2)):"
    )
    with pd.option_context("display.max_rows", len(budget.summary), "display.width", 200):
        print(budget.summary.to_string(index=False))
    print(f"\nSaved summary ({len(budget.summary)} rows): {summary_path}")


if __name__ == "__main__":
    main()
