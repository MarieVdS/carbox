"""
Uncertainty propagation for CSE (and static) chemistry runs.

Two independent sources of uncertainty on a predicted abundance:

- **rate coefficients** -- each reaction's ``k_j`` is uncertain (for UMIST,
  the A-E accuracy class parsed into ``uncertainty_factor``);
- **parent abundances** -- the initial abundance ``x_k(0)`` of each parent
  species is uncertain (observationally, or by assumption).

Both are propagated the same way: differentiate a full ODE solve and
combine the derivative with the assumed input spread. The solve is
differentiated with :func:`jax.jacrev` (reverse mode -- diffrax's default
adjoint is a ``custom_vjp``). Its cost scales with the number of *output*
components (``species x snapshots``), **not** with the number of inputs, so

- every reaction is differentiated against for free, and
- adding the parent-abundance Jacobian to a rate-coefficient run is free too
  (one extra elementwise term in the same backward pass).

Knobs, both nominal 1:

- ``a_j``: multiplies rate coefficient ``j`` (``a_j k_j``); ``d/d a_j`` at 1
  is ``d/d ln k_j``.
- ``m_k``: multiplies the initial abundance vector entry ``k``
  (``y0 -> y0 * m``); ``d/d m_k`` at 1 is ``d/d ln x_k(0)``.

The first-order "uncertainty shift" of output ``x_i`` from input ``theta_j``
is ``(d x_i / d ln theta_j) * ln(factor_j)`` -- an absolute shift in the
abundance. Per-(species, snapshot) the shifts are combined in quadrature
(independent inputs) into ``sigma``; the two sources add in quadrature into
``sigma_total``. ``relative_uncertainty = sigma_total / nominal_abundance``.

The quadrature sums in :func:`uncertainty_budget`'s summary are always over
the *full* set of reactions and parents. ``reaction_ids`` / ``parents`` only
filter the returned detail tables -- use them to inspect one contributor
without the summary silently becoming a partial sum.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import SimulationConfig
from .network import JNetwork, Network
from .solver import SPY, get_time_grid, solve_network

SnapshotIndex = Union[int, str]

DEFAULT_PARENT_UNCERTAINTY = 1.5


class UncertaintyBudget(NamedTuple):
    """Result of :func:`uncertainty_budget`.

    ``rates`` / ``parents`` are the per-contributor detail tables (``None``
    if that source was not requested); ``summary`` is one row per
    ``(species, snapshot)`` with the full-set quadrature error bars.
    """

    rates: Optional[pd.DataFrame]
    parents: Optional[pd.DataFrame]
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


def _solve_jacobians(
    network, jnetwork, y0, config, species_idx, snapshot_index, want_rates, want_parents
):
    """Nominal abundances + requested Jacobians, from a single solve.

    Shapes: nominal ``(n_out, n_sel)``; ``jac_a`` ``(n_out, n_sel,
    n_reactions)``; ``jac_m`` ``(n_out, n_sel, n_species)``.
    ``n_out`` is ``n_snapshots`` for ``snapshot_index="all"`` else 1.
    """
    n_reactions = int(jnetwork.reactions_number)
    n_species = len(network.species)
    a0 = jnp.ones(n_reactions)
    b0 = jnp.zeros(n_reactions)
    m0 = jnp.ones(n_species)
    all_snapshots = snapshot_index == "all"

    def forward(a, m):
        solution = solve_network(jnetwork, y0 * m, config, rate_modifiers=(a, b0))
        ys = solution.ys[:, species_idx]
        return ys if all_snapshots else ys[snapshot_index][None, :]

    nominal = np.asarray(forward(a0, m0))

    if want_rates and want_parents:
        jac_a, jac_m = jax.jacrev(forward, argnums=(0, 1))(a0, m0)
        return nominal, np.asarray(jac_a), np.asarray(jac_m)
    if want_rates:
        return nominal, np.asarray(jax.jacrev(forward, argnums=0)(a0, m0)), None
    return nominal, None, np.asarray(jax.jacrev(forward, argnums=1)(a0, m0))


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

    nominal, jac_a, _ = _solve_jacobians(
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
    nominal, _, jac_m = _solve_jacobians(
        network, jnetwork, y0, config, species_idx, snapshot_index,
        want_rates=False, want_parents=True,
    )
    return _parent_detail(
        parent_list, factors, jac_m, nominal, y0, species_names, times, radius,
        parents, min_abs_shift,
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
    parent_uncertainty: Optional[Dict[str, float]] = None,
    parent_uncertainty_default: float = DEFAULT_PARENT_UNCERTAINTY,
    min_abs_shift: float = 0.0,
) -> UncertaintyBudget:
    """
    Propagate rate-coefficient and/or parent-abundance uncertainty to a
    per-species error bar, from a single solve.

    Parameters
    ----------
    sources
        Which uncertainty sources to include: ``"rates"``, ``"parents"``, or
        both (default).
    reaction_ids, parents
        Inspection filters for the returned ``rates`` / ``parents`` detail
        tables. They do **not** affect ``summary`` -- its ``sigma_*`` columns
        are always the quadrature sum over every reaction / every parent.
    parent_uncertainty, parent_uncertainty_default
        ``{name: factor}`` map (optional ``"default"`` key) for parent
        spreads; falls back to ``config.parent_uncertainties`` then the
        scalar default.

    Returns
    -------
    UncertaintyBudget
        ``rates`` / ``parents`` detail tables (``None`` when not requested)
        and a ``summary`` with one row per ``(species, snapshot)``:
        ``nominal_abundance, sigma_from_rates, sigma_from_parents,
        sigma_total, worst_case_from_rates, worst_case_from_parents,
        n_reactions, n_parents, relative_uncertainty``.
    """
    want_rates = "rates" in sources
    want_parents = "parents" in sources
    if not (want_rates or want_parents):
        raise ValueError("sources must include 'rates' and/or 'parents'")

    species_names = _resolve_species(network, species)
    species_idx = jnp.array(network.get_indexes(species_names))
    times, radius = _snapshot_axes(config, snapshot_index)
    n_out = len(times)

    parent_list, factors = _resolve_parents(
        network, y0, config, parent_uncertainty, parent_uncertainty_default
    )

    nominal, jac_a, jac_m = _solve_jacobians(
        network, jnetwork, y0, config, species_idx, snapshot_index,
        want_rates=want_rates, want_parents=want_parents,
    )

    # --- full-set quadrature error bars (never filtered) ---
    zeros = np.zeros((n_out, len(species_names)))
    sigma_rates, worst_rates = (zeros, zeros)
    sigma_parents, worst_parents = (zeros, zeros)

    if want_rates:
        ln_f = _reaction_ln_factors(network)  # (n_reactions,)
        sigma_rates, worst_rates = _quadrature(jac_a * ln_f[None, None, :])
    if want_parents and parent_list:
        cols = [pidx for _, pidx in parent_list]
        ln_g = np.array([np.log(factors[name]) for name, _ in parent_list])
        sigma_parents, worst_parents = _quadrature(jac_m[:, :, cols] * ln_g[None, None, :])

    sigma_total = np.sqrt(sigma_rates**2 + sigma_parents**2)

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
                    "sigma_total": tot,
                    "worst_case_from_rates": float(worst_rates[o, si]) if want_rates else np.nan,
                    "worst_case_from_parents": float(worst_parents[o, si]) if want_parents else np.nan,
                    "n_reactions": len(network.reactions) if want_rates else 0,
                    "n_parents": len(parent_list) if want_parents else 0,
                    "relative_uncertainty": tot / nom if nom > 0 else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)

    # --- detail tables (filterable) ---
    rates_df = None
    parents_df = None
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

    return UncertaintyBudget(rates=rates_df, parents=parents_df, summary=summary)


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
