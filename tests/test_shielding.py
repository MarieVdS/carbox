"""
Tests for carbox.shielding: CO/C photo reactions get rewritten to
self-shielded rate terms, self-shielding preserves CO deep in a CSE
outflow where the dust-only rate would destroy it, and the two CO methods
(vdb table / one-band) both run and roughly agree.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carbox import SimulationConfig, configure_self_shielding  # noqa: E402
from carbox.initial_conditions import initialize_abundances  # noqa: E402
from carbox.parsers import parse_chemical_network  # noqa: E402
from carbox.physics import CSEPhysics, StaticCloudPhysics  # noqa: E402
from carbox.reactions import (  # noqa: E402
    CIonizationReaction,
    COPhotoDissReaction,
    UMISTPhotoReaction,
)
from carbox.solver import solve_network  # noqa: E402

PHOTO_NET = PROJECT_ROOT / "tests" / "test_data" / "test_umist_photo.csv"
IC = {"H2": 0.5, "CO": 1.5e-4, "O": 2.0e-4, "C": 1.0e-4, "e-": 1.0e-8}


def _cse_config(**overrides):
    physics = CSEPhysics(
        mdot=1e-5, vexp=15.0e5, t_star=2000.0, r_init=1e16, r_star=5e13, eps=0.7
    )
    t_end = (1.1e17 - physics.r_init) / physics.vexp / 3.15576e7
    kw = dict(
        cr_rate=1.0,
        fuv_field=1.0,
        t_start=0.0,
        t_end=t_end,
        n_snapshots=40,
        rtol=1e-8,
        atol=1e-22,
        solver="kvaerno5",
        linear_solver="lu",
        max_steps=200000,
        initial_abundances=IC,
        physics_model=physics,
        run_name="shield_test",
    )
    kw.update(overrides)
    return SimulationConfig(**kw)


def _solve_co(config):
    network = parse_chemical_network(str(PHOTO_NET), "umist")
    configure_self_shielding(network, config)
    y0 = initialize_abundances(network, config)
    jnetwork = network.get_ode()
    solution = solve_network(jnetwork, y0, config)
    return network, np.asarray(solution.ys[:, network.get_index("CO")])


def test_only_co_rewritten_by_default():
    config = _cse_config()
    network = parse_chemical_network(str(PHOTO_NET), "umist")
    assert any(isinstance(r, UMISTPhotoReaction) for r in network.reactions)

    configure_self_shielding(network, config)

    classes = {type(r) for r in network.reactions}
    assert COPhotoDissReaction in classes
    assert CIonizationReaction not in classes  # C photoionization left alone
    assert network._n_self_shielded == 1
    # the C photo reaction stays a plain dust-only UMISTPhotoReaction
    c_photo = next(
        r for r in network.reactions
        if isinstance(r, UMISTPhotoReaction) and set(r.reactants) == {"C"}
    )
    assert set(c_photo.products) == {"C+", "e-"}

    co = next(r for r in network.reactions if isinstance(r, COPhotoDissReaction))
    assert co.reaction_id == 8259  # metadata carried across the swap
    assert co.column_scale > 0.0  # CSE geometry wired in


def test_c_ionization_rewritten_when_opted_in():
    config = _cse_config(shield_c_ionization=True)
    network = parse_chemical_network(str(PHOTO_NET), "umist")
    configure_self_shielding(network, config)

    classes = {type(r) for r in network.reactions}
    assert COPhotoDissReaction in classes
    assert CIonizationReaction in classes
    assert network._n_self_shielded == 2
    ci = next(r for r in network.reactions if isinstance(r, CIonizationReaction))
    assert ci.column_scale > 0.0


def test_self_shielding_preserves_co():
    _, co_off = _solve_co(_cse_config(self_shielding=False))
    _, co_vdb = _solve_co(_cse_config(self_shielding=True, co_shielding_method="vdb"))

    # Dust-only: CO is photodissociated away in the outer envelope.
    assert co_off[-1] < 0.1 * co_off[0]
    # With self-shielding most of the CO survives to the outer edge...
    assert co_vdb[-1] > 0.5 * co_vdb[0]
    # ... and shielding never destroys CO faster than dust alone.
    assert np.all(co_vdb >= co_off - 1e-25)


def test_oneband_matches_vdb_within_factor():
    _, co_vdb = _solve_co(_cse_config(co_shielding_method="vdb"))
    _, co_ob = _solve_co(_cse_config(co_shielding_method="oneband"))

    # Same qualitative behaviour; the two prescriptions differ but not wildly.
    ratio = co_ob[-1] / co_vdb[-1]
    assert 0.5 < ratio < 2.0


def test_oneband_requires_outflow():
    config = SimulationConfig(
        number_density=1e4,
        temperature=30.0,
        initial_abundances=IC,
        t_end=1e4,
        self_shielding=True,
        co_shielding_method="oneband",
        physics_model=StaticCloudPhysics(number_density=1e4, temperature=30.0),
    )
    network = parse_chemical_network(str(PHOTO_NET), "umist")
    with pytest.raises(ValueError, match="oneband"):
        configure_self_shielding(network, config)


def test_static_cloud_unaffected_by_default_vdb():
    """A static-cloud UMIST run still rewrites to vdb shielding but with
    column_scale = 0, i.e. the fixed cloud_radius_pc path length."""
    config = SimulationConfig(
        number_density=1e4,
        temperature=30.0,
        cloud_radius_pc=0.1,
        initial_abundances=IC,
        t_end=1e4,
        physics_model=StaticCloudPhysics(
            number_density=1e4, temperature=30.0, cloud_radius_pc=0.1
        ),
    )
    network = parse_chemical_network(str(PHOTO_NET), "umist")
    configure_self_shielding(network, config)
    co = next(r for r in network.reactions if isinstance(r, COPhotoDissReaction))
    assert co.column_scale == 0.0
