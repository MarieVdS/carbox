#!/usr/bin/env python3
"""
Compare CO self-shielding treatments in a CSE outflow.

Runs the same outflow three ways -- dust-only (no self-shielding), van
Dishoeck & Black (1988) table shielding, and the Morris & Jura (1983)
one-band approximation -- and writes a CO(radius) comparison table + plot.

Example
-------
    python compare_shielding.py --network umist --species CO
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from carbox.initial_conditions import initialize_abundances
from carbox.parsers import parse_chemical_network
from carbox.physics import SPY
from carbox.shielding import configure_self_shielding
from carbox.solver import solve_network

from run_cse import add_common_cse_args, build_cse_config

CASES = {
    "dust_only": dict(self_shielding=False, co_shielding_method="vdb"),
    "vdb": dict(self_shielding=True, co_shielding_method="vdb"),
    "oneband": dict(self_shielding=True, co_shielding_method="oneband"),
}


def _run_case(base_kwargs, case_kwargs):
    input_file, fmt, run_name, output_dir, config = build_cse_config(
        save_derivatives=False, save_rates=False, verbose=False,
        **{**base_kwargs, **case_kwargs},
    )
    network = parse_chemical_network(str(input_file), fmt)
    configure_self_shielding(network, config)
    y0 = initialize_abundances(network, config)
    jnetwork = network.get_ode()
    solution = solve_network(jnetwork, y0, config)
    radius = np.asarray(
        [float(config.physics_model.get_conditions(t)[3]) for t in solution.ts]
    )
    return {
        "config": config,
        "network": network,
        "ts": np.asarray(solution.ts),
        "radius": radius,
        "ys": np.asarray(solution.ys),
        "output_dir": output_dir,
        "run_name": run_name,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="results")
    parser.add_argument("--species", nargs="+", default=["CO"],
                        help="Species to plot (default: CO)")
    add_common_cse_args(parser)
    args = vars(parser.parse_args())
    species = args.pop("species")
    # these are per-case, not global
    args.pop("self_shielding", None)
    args.pop("co_shielding_method", None)

    frames = []
    series = {}
    for case, case_kwargs in CASES.items():
        res = _run_case(args, case_kwargs)
        series[case] = res
        for sp in species:
            idx = res["network"].get_index(sp)
            frames.append(pd.DataFrame({
                "case": case,
                "species": sp,
                "time_years": res["ts"] / SPY,
                "radius_cm": res["radius"],
                "abundance": res["ys"][:, idx],
            }))

    output_dir = series["vdb"]["output_dir"]
    run_name = series["vdb"]["run_name"]
    table = pd.concat(frames, ignore_index=True)
    out_csv = output_dir / f"{run_name}_shielding_comparison.csv"
    table.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    fig, axes = plt.subplots(
        1, len(species), figsize=(5 * len(species), 4), squeeze=False
    )
    for ax, sp in zip(axes[0], species):
        for case in CASES:
            res = series[case]
            ax.loglog(res["radius"], res["ys"][:, res["network"].get_index(sp)], label=case)
        ax.set_xlabel("radius [cm]")
        ax.set_ylabel(f"x({sp})")
        ax.set_title(sp)
        ax.legend()
    fig.tight_layout()
    out_png = output_dir / f"{run_name}_shielding_comparison.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")

    # Quick numeric summary at the outer edge
    print("\nFinal-radius abundances:")
    print(
        table[table["radius_cm"] == table["radius_cm"].max()]
        .pivot_table(index="species", columns="case", values="abundance")
        .to_string()
    )


if __name__ == "__main__":
    main()
