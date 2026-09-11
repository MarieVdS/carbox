"""
Uncertainty propagation for CSE (and static) chemistry runs.

Three independent sources of uncertainty on a predicted abundance:

- **rate coefficients** -- each reaction's ``k_j`` is uncertain (for UMIST,
  the A-E accuracy class parsed into ``uncertainty_factor``);
- **parent abundances** -- the initial abundance ``x_k(0)`` of each parent
  species is uncertain (observationally, or by assumption);
- **physical parameters** -- the physics-model parameters that set the
  density / temperature / geometry profiles are uncertain. For a CSE outflow
  these are ``mdot``, ``vexp``, ``t_star`` and ``eps`` (opt-in;
  ``sources`` must include ``"physics"``).

All are propagated the same way: differentiate a full ODE solve and
combine the derivative with the assumed input spread. The solve is
differentiated with :func:`jax.jacrev` (reverse mode -- diffrax's default
adjoint, ``RecursiveCheckpointAdjoint``, is a ``custom_vjp``). Its cost
scales with the number of *output* components (``species x snapshots``),
**not** with the number of inputs, so every reaction / parent / physics
parameter is differentiated against in the same backward pass.

Knobs, all nominal 1:

- ``a_j``: multiplies rate coefficient ``j`` (``a_j k_j``); ``d/d a_j`` at 1
  is ``d/d ln k_j``.
- ``m_k``: multiplies the initial abundance vector entry ``k``
  (``y0 -> y0 * m``); ``d/d m_k`` at 1 is ``d/d ln x_k(0)``.
- ``p_l``: multiplies physics-model parameter ``l`` (``theta_l -> theta_l *
  p_l``); ``d/d p_l`` at 1 is ``d/d ln theta_l``. When ``vexp`` is perturbed
  ``t_end`` is recomputed so the outflow still reaches the same outer radius
  (a **fixed-radius** comparison; with ``t_start = 0`` the perturbed run also
  lands on the same log-radius snapshot grid).

The first-order "uncertainty shift" of output ``x_i`` from input ``theta_j``
is ``(d x_i / d ln theta_j) * ln(factor_j)`` -- an absolute shift in the
abundance. Per-(species, snapshot) the shifts are combined in quadrature
(independent inputs) into ``sigma``; the sources add in quadrature into
``sigma_total``. ``relative_uncertainty = sigma_total / nominal_abundance``.
This is a first-order (local) linearisation -- fine for percent-level
spreads, only approximate for factor-of-several excursions (e.g. an
order-of-magnitude ``mdot``), where a model grid is more faithful.

The quadrature sums in :func:`uncertainty_budget`'s summary are always over
the *full* set of reactions / parents / physics parameters.
``reaction_ids`` / ``parents`` / ``physics_params`` only filter the returned
detail tables -- use them to inspect one contributor without the summary
silently becoming a partial sum.
"""

import dataclasses
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import SimulationConfig
from .network import JNetwork, Network
from .solver import SPY, get_time_grid, solve_network

SnapshotIndex = Union[int, str]

DEFAULT_PARENT_UNCERTAINTY = 1.5
DEFAULT_PHYSICS_UNCERTAINTY = 1.0

# Physics-model parameters exposed to physical-parameter sensitivity, per
# model class. Only fields that are plain float leaves of the model.
PHYSICS_PARAMS_BY_MODEL: Dict[str, tuple] = {
    "CSEPhysics": ("mdot", "vexp", "t_star", "eps"),
}


class UncertaintyBudget(NamedTuple):
    """Result of :func:`uncertainty_budget`.

    ``rates`` / ``parents`` / ``physics`` are the per-contributor detail
    tables (``None`` if that source was not requested); ``summary`` is one
    row per ``(species, snapshot)`` with the full-set quadrature error bars.
    """

    rates: Optional[pd.DataFrame]
    parents: Optional[pd.DataFrame]
    physics: Optional[pd.DataFrame]
    summary: pd.DataFrame


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------
def _reaction_column_index(network: Network) -> dict:
    """``reaction_id -> j`` in the rate / ``rate_modifier`` vector.

    ``Network.get_ode()`` reorders ``network.reactions`` in place to match
    column ``j`` of the incidence matrix; call this only afterwards.
    """
    return {reaction.reaction_id: j for j, reaction in enumerate(network.reactions)}


def _resolve_species(network: Network, species: Optional[Sequence[str]]) -> List[str]:
    names = list(species) if species is not None else [s.name for s in network.species]
    try:
        network.get_indexes(names)
    except ValueError as exc:  # pragma: no cover - message passthrough
        raise ValueError(
            f"Unknown species requested ({exc}). "
            f"Available: {[s.name for s in network.species]}"
        ) from exc
    return names


def _snapshot_axes(config: SimulationConfig, snapshot_index: SnapshotIndex):
    """(selected_times [s], radius [cm]) as numpy arrays."""
    t_all = get_time_grid(config)
    selected = (
        t_all if snapshot_index == "all" else jnp.atleast_1d(t_all[snapshot_index])
    )
    _, _, _, radius = jax.vmap(config.physics_model.get_conditions)(selected)
    return np.asarray(selected), np.asarray(radius)


def _apply_physics_multipliers(
    config: SimulationConfig, p: jnp.ndarray, names: Sequence[str]
) -> SimulationConfig:
    """Config copy with physics-model field ``theta_l -> theta_l * p[l]``.

    Holds the outer radius fixed: if ``vexp`` is among ``names``, ``t_end`` is
    recomputed so the outflow still ends at the nominal ``r_final``. With
    ``t_start = 0`` this also keeps the perturbed run on the same log-radius
    snapshot grid (:meth:`CSEPhysics.time_grid`), so a snapshot maps to the
    same radius in the perturbed and nominal solves.
    """
    physics = config.physics_model
    new_values = [getattr(physics, nm) * p[i] for i, nm in enumerate(names)]
    physics2 = eqx.tree_at(
        lambda ph: [getattr(ph, nm) for nm in names], physics, new_values
    )

    t_end = config.t_end
    if "vexp" in names:
        # r_final is constant w.r.t. p: r_final = r_init + vexp_cgs * t_end_sec.
        # vexp is already in cm/s (CSEPhysics convention).
        r_final = physics.r_init + physics.vexp * (config.t_end * SPY)
        t_end = (r_final - physics.r_init) / physics2.vexp / SPY

    return dataclasses.replace(config, physics_model=physics2, t_end=t_end)


def _solve_jacobians(
    network,
    jnetwork,
    y0,
    config,
    species_idx,
    snapshot_index,
    want_rates,
    want_parents,
    want_physics=False,
    physics_names=(),
):
    """Nominal abundances + requested Jacobians, from a single solve.

    Shapes: nominal ``(n_out, n_sel)``; ``jac_a`` ``(n_out, n_sel,
    n_reactions)``; ``jac_m`` ``(n_out, n_sel, n_species)``; ``jac_p``
    ``(n_out, n_sel, len(physics_names))``. Unrequested Jacobians are
    ``None``. ``n_out`` is ``n_snapshots`` for ``snapshot_index="all"`` else 1.
    """
    n_reactions = int(jnetwork.reactions_number)
    n_species = len(network.species)
    a0 = jnp.ones(n_reactions)
    b0 = jnp.zeros(n_reactions)
    m0 = jnp.ones(n_species)
    p0 = jnp.ones(len(physics_names))
    all_snapshots = snapshot_index == "all"

    def forward(a, m, p):
        cfg = (
            _apply_physics_multipliers(config, p, physics_names)
            if want_physics
            else config
        )
        solution = solve_network(jnetwork, y0 * m, cfg, rate_modifiers=(a, b0))
        ys = solution.ys[:, species_idx]
        return ys if all_snapshots else ys[snapshot_index][None, :]

    nominal = np.asarray(forward(a0, m0, p0))

    argnums = []
    if want_rates:
        argnums.append(0)
    if want_parents:
        argnums.append(1)
    if want_physics:
        argnums.append(2)

    jac_a = jac_m = jac_p = None
    if argnums:
        # argnums as a tuple -> jacrev returns a tuple (even for length 1)
        jacs = jax.jacrev(forward, argnums=tuple(argnums))(a0, m0, p0)
        by_arg = dict(zip(argnums, jacs))
        if 0 in by_arg:
            jac_a = np.asarray(by_arg[0])
        if 1 in by_arg:
            jac_m = np.asarray(by_arg[1])
        if 2 in by_arg:
            jac_p = np.asarray(by_arg[2])
    return nominal, jac_a, jac_m, jac_p


def _reaction_ln_factors(network: Network) -> np.ndarray:
    """``ln(uncertainty_factor)`` per reaction column (0 if no info)."""
    return np.array(
        [
            float(np.log(float(getattr(r, "uncertainty_factor", 1.0) or 1.0)))
            for r in network.reactions
        ]
    )


def _resolve_parents(
    network: Network,
    y0: jnp.ndarray,
    config: SimulationConfig,
    parent_uncertainty: Optional[Dict[str, float]],
    default_factor: float,
):
    """[(name, species_index)], {name: factor} for the uncertain parents.

    Parents = species named in ``config.initial_abundances`` that are in the
    network and start above the abundance floor.
    """
    names = [s.name for s in network.species]
    y0_np = np.asarray(y0)
    floor = getattr(config, "abundance_floor", 1e-30)

    parents = []
    for name in getattr(config, "initial_abundances", {}) or {}:
        if name in names:
            idx = names.index(name)
            if y0_np[idx] > floor:
                parents.append((name, idx))

    source: Dict[str, float] = {}
    if parent_uncertainty is not None:
        source = dict(parent_uncertainty)
    elif getattr(config, "parent_uncertainties", None):
        source = dict(config.parent_uncertainties)
    default_factor = float(source.pop("default", default_factor))

    factors = {name: float(source.get(name, default_factor)) for name, _ in parents}
    return parents, factors


def _resolve_physics_params(
    config: SimulationConfig,
    physics_uncertainty: Optional[Dict[str, float]],
    default_factor: float,
):
    """[(name, nominal_value)], {name: factor} for the physics parameters.

    Parameter list comes from :data:`PHYSICS_PARAMS_BY_MODEL` keyed on the
    physics-model class name; factors from the explicit ``physics_uncertainty``
    map, else ``config.physics_uncertainties``, else the scalar default (with
    an optional ``"default"`` key overriding it).
    """
    physics = getattr(config, "physics_model", None)
    model_name = type(physics).__name__
    names = PHYSICS_PARAMS_BY_MODEL.get(model_name)
    if names is None:
        raise ValueError(
            f"Physical-parameter sensitivity is not defined for "
            f"physics_model={model_name!r}. Supported models: "
            f"{sorted(PHYSICS_PARAMS_BY_MODEL)}."
        )
    params = [(nm, float(getattr(physics, nm))) for nm in names]

    source: Dict[str, float] = {}
    if physics_uncertainty is not None:
        source = dict(physics_uncertainty)
    elif getattr(config, "physics_uncertainties", None):
        source = dict(config.physics_uncertainties)
    default_factor = float(source.pop("default", default_factor))

    factors = {nm: float(source.get(nm, default_factor)) for nm, _ in params}
    return params, factors


# --------------------------------------------------------------------------
# detail tables
# --------------------------------------------------------------------------
def _rate_detail(
    network, jac_a, nominal, species_names, times, radius, reaction_ids, min_abs_shift
) -> pd.DataFrame:
    column_of = _reaction_column_index(network)
    if reaction_ids is None:
        ids = list(column_of)
    else:
        missing = [r for r in reaction_ids if r not in column_of]
        if missing:
            raise ValueError(f"reaction_id(s) not in this network: {missing}")
        ids = list(reaction_ids)

    by_id = {r.reaction_id: r for r in network.reactions}
    rows = []
    for rid in ids:
        j = column_of[rid]
        rx = by_id[rid]
        factor = float(getattr(rx, "uncertainty_factor", 1.0) or 1.0)
        ln_factor = float(np.log(factor))
        rx_str = f"{' + '.join(rx.reactants)} -> {' + '.join(rx.products)}"
        for o in range(jac_a.shape[0]):
            for si, sp in enumerate(species_names):
                sens = float(jac_a[o, si, j])
                shift = sens * ln_factor
                if abs(shift) < min_abs_shift:
                    continue
                rows.append(
                    {
                        "reaction_id": rid,
                        "reaction_type": rx.reaction_type,
                        "reaction": rx_str,
                        "uncertainty_flag": getattr(rx, "uncertainty_flag", None),
                        "uncertainty_factor": factor,
                        "species": sp,
                        "time_years": float(times[o]) / SPY,
                        "radius_cm": float(radius[o]),
                        "nominal_abundance": float(nominal[o, si]),
                        "d_abundance_d_lnk": sens,
                        "uncertainty_shift": shift,
                    }
                )
    return pd.DataFrame(rows)


def _parent_detail(
    parents, factors, jac_m, nominal, y0, species_names, times, radius, parent_filter,
    min_abs_shift,
) -> pd.DataFrame:
    y0_np = np.asarray(y0)
    wanted = set(parent_filter) if parent_filter is not None else None
    if wanted is not None:
        unknown = wanted - {name for name, _ in parents}
        if unknown:
            raise ValueError(f"parent(s) not an uncertain parent in this run: {sorted(unknown)}")

    rows = []
    for name, pidx in parents:
        if wanted is not None and name not in wanted:
            continue
        ln_factor = float(np.log(factors[name]))
        for o in range(jac_m.shape[0]):
            for si, sp in enumerate(species_names):
                sens = float(jac_m[o, si, pidx])
                shift = sens * ln_factor
                if abs(shift) < min_abs_shift:
                    continue
                rows.append(
                    {
                        "parent": name,
                        "parent_abundance": float(y0_np[pidx]),
                        "uncertainty_factor": factors[name],
                        "species": sp,
                        "time_years": float(times[o]) / SPY,
                        "radius_cm": float(radius[o]),
                        "nominal_abundance": float(nominal[o, si]),
                        "d_abundance_d_lnx0": sens,
                        "uncertainty_shift": shift,
                    }
                )
    return pd.DataFrame(rows)


def _physics_detail(
    params, factors, jac_p, nominal, species_names, times, radius, param_filter,
    min_abs_shift,
) -> pd.DataFrame:
    wanted = set(param_filter) if param_filter is not None else None
    if wanted is not None:
        unknown = wanted - {name for name, _ in params}
        if unknown:
            raise ValueError(
                f"physics parameter(s) not in this run: {sorted(unknown)}"
            )

    rows = []
    for li, (name, nominal_value) in enumerate(params):
        if wanted is not None and name not in wanted:
            continue
        ln_factor = float(np.log(factors[name]))
        for o in range(jac_p.shape[0]):
            for si, sp in enumerate(species_names):
                sens = float(jac_p[o, si, li])
                shift = sens * ln_factor
                if abs(shift) < min_abs_shift:
                    continue
                rows.append(
                    {
                        "parameter": name,
                        "nominal_value": nominal_value,
                        "uncertainty_factor": factors[name],
                        "species": sp,
                        "time_years": float(times[o]) / SPY,
                        "radius_cm": float(radius[o]),
                        "nominal_abundance": float(nominal[o, si]),
                        "d_abundance_d_lnparam": sens,
                        "uncertainty_shift": shift,
                    }
                )
    return pd.DataFrame(rows)


def _quadrature(contrib: np.ndarray):
    """(sigma, worst_case) over the last axis of a (n_out, n_sel, n_in) array."""
    sigma = np.sqrt((contrib**2).sum(axis=2))
    worst = np.abs(contrib).sum(axis=2)
    return sigma, worst


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def rate_coefficient_sensitivity(
    network: Network,
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
    *,
    species: Optional[Sequence[str]] = None,
    reaction_ids: Optional[Sequence[int]] = None,
    snapshot_index: SnapshotIndex = -1,
    min_abs_shift: float = 0.0,
) -> pd.DataFrame:
    """Detail table of ``d(abundance)/d ln k`` per (reaction, species, snapshot).

    See the module docstring. ``reaction_ids`` filters the returned rows
    only; ``species`` and ``snapshot_index`` are the cost levers. For the
    combined error budget use :func:`uncertainty_budget`.
    """
    species_names = _resolve_species(network, species)
    species_idx = jnp.array(network.get_indexes(species_names))
    times, radius = _snapshot_axes(config, snapshot_index)

    nominal, jac_a, _, _ = _solve_jacobians(
        network, jnetwork, y0, config, species_idx, snapshot_index,
        want_rates=True, want_parents=False,
    )
    return _rate_detail(
        network, jac_a, nominal, species_names, times, radius, reaction_ids, min_abs_shift
    )


def initial_abundance_sensitivity(
    network: Network,
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
    *,
    species: Optional[Sequence[str]] = None,
    parents: Optional[Sequence[str]] = None,
    snapshot_index: SnapshotIndex = -1,
    parent_uncertainty: Optional[Dict[str, float]] = None,
    parent_uncertainty_default: float = DEFAULT_PARENT_UNCERTAINTY,
    min_abs_shift: float = 0.0,
) -> pd.DataFrame:
    """Detail table of ``d(abundance)/d ln x_parent(0)`` per (parent, species,
    snapshot).

    ``parents`` filters the returned rows only. ``parent_uncertainty`` is a
    ``{name: factor}`` map (with an optional ``"default"`` key); falls back
    to ``config.parent_uncertainties`` then ``parent_uncertainty_default``.
    """
    species_names = _resolve_species(network, species)
    species_idx = jnp.array(network.get_indexes(species_names))
    times, radius = _snapshot_axes(config, snapshot_index)

    parent_list, factors = _resolve_parents(
        network, y0, config, parent_uncertainty, parent_uncertainty_default
    )
    nominal, _, jac_m, _ = _solve_jacobians(
        network, jnetwork, y0, config, species_idx, snapshot_index,
        want_rates=False, want_parents=True,
    )
    return _parent_detail(
        parent_list, factors, jac_m, nominal, y0, species_names, times, radius,
        parents, min_abs_shift,
    )


def physical_parameter_sensitivity(
    network: Network,
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
    *,
    species: Optional[Sequence[str]] = None,
    physics_params: Optional[Sequence[str]] = None,
    snapshot_index: SnapshotIndex = -1,
    physics_uncertainty: Optional[Dict[str, float]] = None,
    physics_uncertainty_default: float = DEFAULT_PHYSICS_UNCERTAINTY,
    min_abs_shift: float = 0.0,
) -> pd.DataFrame:
    """Detail table of ``d(abundance)/d ln theta`` per (physics parameter,
    species, snapshot).

    ``theta`` ranges over the physics-model parameters
    (:data:`PHYSICS_PARAMS_BY_MODEL`; ``mdot``/``vexp``/``t_star``/``eps`` for
    a CSE). ``vexp`` uses the fixed-radius convention (see the module
    docstring). ``physics_params`` filters the returned rows only.
    """
    species_names = _resolve_species(network, species)
    species_idx = jnp.array(network.get_indexes(species_names))
    times, radius = _snapshot_axes(config, snapshot_index)

    param_list, factors = _resolve_physics_params(
        config, physics_uncertainty, physics_uncertainty_default
    )
    nominal, _, _, jac_p = _solve_jacobians(
        network, jnetwork, y0, config, species_idx, snapshot_index,
        want_rates=False, want_parents=False,
        want_physics=True, physics_names=tuple(nm for nm, _ in param_list),
    )
    return _physics_detail(
        param_list, factors, jac_p, nominal, species_names, times, radius,
        physics_params, min_abs_shift,
    )


def uncertainty_budget(
    network: Network,
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
    *,
    species: Optional[Sequence[str]] = None,
    snapshot_index: SnapshotIndex = -1,
    sources: Sequence[str] = ("rates", "parents"),
    reaction_ids: Optional[Sequence[int]] = None,
    parents: Optional[Sequence[str]] = None,
    physics_params: Optional[Sequence[str]] = None,
    parent_uncertainty: Optional[Dict[str, float]] = None,
    parent_uncertainty_default: float = DEFAULT_PARENT_UNCERTAINTY,
    physics_uncertainty: Optional[Dict[str, float]] = None,
    physics_uncertainty_default: float = DEFAULT_PHYSICS_UNCERTAINTY,
    min_abs_shift: float = 0.0,
) -> UncertaintyBudget:
    """
    Propagate rate-coefficient, parent-abundance and/or physical-parameter
    uncertainty to a per-species error bar, from a single solve.

    Parameters
    ----------
    sources
        Which uncertainty sources to include: any of ``"rates"``,
        ``"parents"``, ``"physics"`` (default ``("rates", "parents")`` --
        ``"physics"`` is opt-in).
    reaction_ids, parents, physics_params
        Inspection filters for the returned ``rates`` / ``parents`` /
        ``physics`` detail tables. They do **not** affect ``summary`` -- its
        ``sigma_*`` columns are always the quadrature sum over every
        reaction / parent / physics parameter.
    parent_uncertainty, parent_uncertainty_default
        ``{name: factor}`` map (optional ``"default"`` key) for parent
        spreads; falls back to ``config.parent_uncertainties`` then the
        scalar default.
    physics_uncertainty, physics_uncertainty_default
        ``{name: factor}`` map (optional ``"default"`` key) for the physics
        parameters (``mdot``/``vexp``/``t_star``/``eps`` for a CSE); falls
        back to ``config.physics_uncertainties`` then the scalar default
        (``1.0`` -- no uncertainty).

    Returns
    -------
    UncertaintyBudget
        ``rates`` / ``parents`` / ``physics`` detail tables (``None`` when
        not requested) and a ``summary`` with one row per
        ``(species, snapshot)``: ``nominal_abundance, sigma_from_rates,
        sigma_from_parents, sigma_from_physics, sigma_total,
        worst_case_from_{rates,parents,physics}, n_reactions, n_parents,
        n_physics_params, relative_uncertainty``.
    """
    want_rates = "rates" in sources
    want_parents = "parents" in sources
    want_physics = "physics" in sources
    if not (want_rates or want_parents or want_physics):
        raise ValueError(
            "sources must include at least one of 'rates', 'parents', 'physics'"
        )

    species_names = _resolve_species(network, species)
    species_idx = jnp.array(network.get_indexes(species_names))
    times, radius = _snapshot_axes(config, snapshot_index)
    n_out = len(times)

    parent_list, factors = _resolve_parents(
        network, y0, config, parent_uncertainty, parent_uncertainty_default
    )
    if want_physics:
        physics_list, physics_factors = _resolve_physics_params(
            config, physics_uncertainty, physics_uncertainty_default
        )
    else:
        physics_list, physics_factors = [], {}
    physics_names = tuple(nm for nm, _ in physics_list)

    nominal, jac_a, jac_m, jac_p = _solve_jacobians(
        network, jnetwork, y0, config, species_idx, snapshot_index,
        want_rates=want_rates, want_parents=want_parents,
        want_physics=want_physics, physics_names=physics_names,
    )

    # --- full-set quadrature error bars (never filtered) ---
    zeros = np.zeros((n_out, len(species_names)))
    sigma_rates, worst_rates = (zeros, zeros)
    sigma_parents, worst_parents = (zeros, zeros)
    sigma_physics, worst_physics = (zeros, zeros)

    if want_rates:
        ln_f = _reaction_ln_factors(network)  # (n_reactions,)
        sigma_rates, worst_rates = _quadrature(jac_a * ln_f[None, None, :])
    if want_parents and parent_list:
        cols = [pidx for _, pidx in parent_list]
        ln_g = np.array([np.log(factors[name]) for name, _ in parent_list])
        sigma_parents, worst_parents = _quadrature(jac_m[:, :, cols] * ln_g[None, None, :])
    if want_physics and physics_list:
        ln_h = np.array([np.log(physics_factors[name]) for name, _ in physics_list])
        sigma_physics, worst_physics = _quadrature(jac_p * ln_h[None, None, :])

    sigma_total = np.sqrt(sigma_rates**2 + sigma_parents**2 + sigma_physics**2)

    summary_rows = []
    for o in range(n_out):
        for si, sp in enumerate(species_names):
            nom = float(nominal[o, si])
            tot = float(sigma_total[o, si])
            summary_rows.append(
                {
                    "species": sp,
                    "time_years": float(times[o]) / SPY,
                    "radius_cm": float(radius[o]),
                    "nominal_abundance": nom,
                    "sigma_from_rates": float(sigma_rates[o, si]) if want_rates else np.nan,
                    "sigma_from_parents": float(sigma_parents[o, si]) if want_parents else np.nan,
                    "sigma_from_physics": float(sigma_physics[o, si]) if want_physics else np.nan,
                    "sigma_total": tot,
                    "worst_case_from_rates": float(worst_rates[o, si]) if want_rates else np.nan,
                    "worst_case_from_parents": float(worst_parents[o, si]) if want_parents else np.nan,
                    "worst_case_from_physics": float(worst_physics[o, si]) if want_physics else np.nan,
                    "n_reactions": len(network.reactions) if want_rates else 0,
                    "n_parents": len(parent_list) if want_parents else 0,
                    "n_physics_params": len(physics_list) if want_physics else 0,
                    "relative_uncertainty": tot / nom if nom > 0 else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)

    # --- detail tables (filterable) ---
    rates_df = None
    parents_df = None
    physics_df = None
    if want_rates:
        rates_df = _rate_detail(
            network, jac_a, nominal, species_names, times, radius,
            reaction_ids, min_abs_shift,
        )
    if want_parents:
        parents_df = _parent_detail(
            parent_list, factors, jac_m, nominal, y0, species_names, times, radius,
            parents, min_abs_shift,
        )
    if want_physics:
        physics_df = _physics_detail(
            physics_list, physics_factors, jac_p, nominal, species_names, times,
            radius, physics_params, min_abs_shift,
        )

    return UncertaintyBudget(
        rates=rates_df, parents=parents_df, physics=physics_df, summary=summary
    )


def summarize_uncertainty(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a detail table (:func:`rate_coefficient_sensitivity` or
    :func:`initial_abundance_sensitivity`) into one error bar per
    ``(species, snapshot)`` by quadrature-summing ``uncertainty_shift``.

    This sums over exactly the rows present -- so on a *filtered* table it
    is a partial sum. For the full, source-separated budget use
    :func:`uncertainty_budget`.
    """
    count_col = "reaction_id" if "reaction_id" in sensitivity.columns else "parent"
    group_cols = ["species", "time_years", "radius_cm"]
    summary = (
        sensitivity.groupby(group_cols)
        .agg(
            nominal_abundance=("nominal_abundance", "first"),
            sigma_abundance=("uncertainty_shift", lambda s: float(np.sqrt((s**2).sum()))),
            worst_case_abundance=("uncertainty_shift", lambda s: float(s.abs().sum())),
            n_inputs=(count_col, "count"),
        )
        .reset_index()
    )
    summary["relative_uncertainty"] = (
        summary["sigma_abundance"] / summary["nominal_abundance"]
    )
    # Legacy alias: this column was 'n_reactions' before parent sensitivity
    # existed and rate-only callers still read it.
    if count_col == "reaction_id":
        summary["n_reactions"] = summary["n_inputs"]
    return summary
