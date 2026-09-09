"""
Tests for carbox.sensitivity: the jacrev-through-solve rate-coefficient
sensitivity must agree with finite differences on the rate_modifier knob,
and the uncertainty summary must combine shifts in quadrature.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carbox import SimulationConfig  # noqa: E402
from carbox.initial_conditions import initialize_abundances  # noqa: E402
from carbox.parsers import parse_chemical_network  # noqa: E402
from carbox.sensitivity import (  # noqa: E402
    initial_abundance_sensitivity,
    rate_coefficient_sensitivity,
    summarize_uncertainty,
    uncertainty_budget,
)
from carbox.solver import solve_network  # noqa: E402

NETWORK = PROJECT_ROOT / "data" / "umist22_mini.csv"
SPECIES = ["CO", "HCO+", "OH", "H2O"]


@pytest.fixture(scope="module")
def prepared():
    config = SimulationConfig(
        number_density=1e6,
        temperature=300.0,
        cr_rate=1.0,
        t_start=0.0,
        t_end=5.0e3,
        n_snapshots=20,
        rtol=1e-8,
        atol=1e-25,
        solver="kvaerno5",
        linear_solver="lu",
        max_steps=65536,
        initial_abundances={"H2": 0.5, "H": 1e-4, "O": 3e-4, "C": 1e-4, "e-": 1e-6},
        run_name="sens_test",
    )
    network = parse_chemical_network(str(NETWORK), "umist")
    y0 = initialize_abundances(network, config)
    jnetwork = network.get_ode()
    return network, jnetwork, y0, config


def test_sensitivity_matches_finite_difference(prepared):
    network, jnetwork, y0, config = prepared

    df = rate_coefficient_sensitivity(
        network, jnetwork, y0, config, species=SPECIES, snapshot_index=-1
    )

    # One row per (reaction, species); no snapshot dimension for a single index
    assert len(df) == len(network.reactions) * len(SPECIES)
    assert set(df["species"]) == set(SPECIES)

    column_of = {r.reaction_id: j for j, r in enumerate(network.reactions)}
    n_reactions = int(jnetwork.reactions_number)
    b_zero = jnp.zeros(n_reactions)

    def final_abundance(a, species_index):
        sol = solve_network(jnetwork, y0, config, rate_modifiers=(a, b_zero))
        return float(sol.ys[-1, species_index])

    # Check the handful of most sensitive (reaction, species) pairs against
    # a central finite difference on the same knob.
    top = df.reindex(df["d_abundance_d_lnk"].abs().sort_values(ascending=False).index)
    checked = 0
    for _, row in top.head(5).iterrows():
        j = column_of[int(row["reaction_id"])]
        sp_index = network.get_index(row["species"])
        a = jnp.ones(n_reactions)
        h = 1e-5
        plus = final_abundance(a.at[j].set(1.0 + h), sp_index)
        minus = final_abundance(a.at[j].set(1.0 - h), sp_index)
        fd = (plus - minus) / (2 * h)
        analytic = row["d_abundance_d_lnk"]
        scale = max(abs(analytic), abs(fd), 1e-30)
        assert abs(analytic - fd) / scale < 1e-4, (
            f"reaction {row['reaction_id']} / {row['species']}: "
            f"analytic {analytic:.6e} vs FD {fd:.6e}"
        )
        checked += 1
    assert checked > 0


def test_uncertainty_summary_quadrature(prepared):
    network, jnetwork, y0, config = prepared

    df = rate_coefficient_sensitivity(
        network, jnetwork, y0, config, species=["CO"], snapshot_index=-1
    )
    summary = summarize_uncertainty(df)

    assert len(summary) == 1
    row = summary.iloc[0]

    expected_sigma = np.sqrt((df["uncertainty_shift"] ** 2).sum())
    expected_worst = df["uncertainty_shift"].abs().sum()
    assert row["n_reactions"] == len(network.reactions)
    assert np.isclose(row["sigma_abundance"], expected_sigma)
    assert np.isclose(row["worst_case_abundance"], expected_worst)
    # quadrature sum never exceeds the worst-case linear sum
    assert row["sigma_abundance"] <= row["worst_case_abundance"] + 1e-30
    assert np.isclose(
        row["relative_uncertainty"], row["sigma_abundance"] / row["nominal_abundance"]
    )


def test_snapshot_all_has_radius_axis(prepared):
    network, jnetwork, y0, config = prepared

    df = rate_coefficient_sensitivity(
        network, jnetwork, y0, config, species=["CO"], snapshot_index="all"
    )
    # one row per (reaction, snapshot) for the single species requested
    assert len(df) == len(network.reactions) * config.n_snapshots
    assert df["time_years"].nunique() == config.n_snapshots


def test_initial_abundance_sensitivity_matches_finite_difference(prepared):
    network, jnetwork, y0, config = prepared

    df = initial_abundance_sensitivity(
        network, jnetwork, y0, config, species=SPECIES, snapshot_index=-1
    )
    # parents = the 5 species set above the floor in initial_abundances
    parents = sorted(df["parent"].unique())
    assert parents == sorted(["H2", "H", "O", "C", "e-"])
    assert len(df) == len(parents) * len(SPECIES)

    names = [s.name for s in network.species]
    y0_np = np.asarray(y0)

    def final_abundance(m, species_index):
        sol = solve_network(jnetwork, y0 * m, config)
        return float(sol.ys[-1, species_index])

    top = df.reindex(df["d_abundance_d_lnx0"].abs().sort_values(ascending=False).index)
    for _, row in top.head(4).iterrows():
        pidx = names.index(row["parent"])
        sp_index = network.get_index(row["species"])
        m = jnp.ones(len(names))
        h = 1e-5
        plus = final_abundance(m.at[pidx].set(1.0 + h), sp_index)
        minus = final_abundance(m.at[pidx].set(1.0 - h), sp_index)
        fd = (plus - minus) / (2 * h)
        analytic = row["d_abundance_d_lnx0"]
        scale = max(abs(analytic), abs(fd), 1e-30)
        assert abs(analytic - fd) / scale < 1e-4
        assert np.isclose(row["parent_abundance"], y0_np[pidx])


def test_uncertainty_budget_combines_sources_in_quadrature(prepared):
    network, jnetwork, y0, config = prepared

    budget = uncertainty_budget(
        network, jnetwork, y0, config, species=SPECIES, snapshot_index=-1,
        parent_uncertainty_default=2.0,
    )
    s = budget.summary
    assert len(s) == len(SPECIES)
    assert budget.rates is not None and budget.parents is not None

    # sigma_total = sqrt(rates^2 + parents^2), elementwise
    np.testing.assert_allclose(
        s["sigma_total"],
        np.sqrt(s["sigma_from_rates"] ** 2 + s["sigma_from_parents"] ** 2),
        rtol=1e-10,
    )
    assert (s["n_reactions"] == len(network.reactions)).all()
    assert (s["n_parents"] == 5).all()
    # each source's summary sigma matches quadrature over its own full detail table
    for sp in SPECIES:
        rate_rows = budget.rates[budget.rates["species"] == sp]
        expected = np.sqrt((rate_rows["uncertainty_shift"] ** 2).sum())
        got = s.loc[s["species"] == sp, "sigma_from_rates"].iloc[0]
        assert np.isclose(got, expected)


def test_budget_detail_filter_does_not_change_summary(prepared):
    network, jnetwork, y0, config = prepared

    full = uncertainty_budget(
        network, jnetwork, y0, config, species=["SiO" if False else "CO"],
        snapshot_index=-1,
    )
    filtered = uncertainty_budget(
        network, jnetwork, y0, config, species=["CO"], snapshot_index=-1,
        parents=["H2"],
    )
    # the parents detail table is filtered...
    assert set(filtered.parents["parent"]) == {"H2"}
    assert set(full.parents["parent"]) != {"H2"}
    # ...but the summary sigma is identical (still the full-set quadrature)
    np.testing.assert_allclose(
        full.summary["sigma_from_parents"], filtered.summary["sigma_from_parents"]
    )
    np.testing.assert_allclose(
        full.summary["sigma_total"], filtered.summary["sigma_total"]
    )
