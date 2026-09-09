"""
Self-shielding for photodissociation / photoionization reactions.

Standard reaction-network files (UMIST, UCLCHEM) tabulate photo rates as
``alpha * chi * exp(-gamma * Av)`` -- dust attenuation only. The line
opacity of CO (shielded by CO and H2) matters enormously in a circumstellar
envelope, where CO survives far past where the dust-only rate would destroy
it.

:func:`configure_self_shielding` rewrites the matching photo reactions in a
parsed network to use the shielded rate terms in :mod:`carbox.reactions`,
wiring in the species indices and, for a CSE run, the outflow geometry:

- ``CO -> O + C`` -> ``COPhotoDissReaction`` -- always, when
  ``config.self_shielding`` is on.
- ``C -> C+ + e-`` -> ``CIonizationReaction`` (UCLCHEM's dust + gas-phase
  treatment) -- **opt-in** via ``config.shield_c_ionization``; off by
  default, since it is not the right prescription for every CSE model.
- ``H2 -> H + H`` -> ``H2PhotoDissReaction`` -- opt-in via
  ``config.shield_h2`` (UMIST networks have no H2 photodissociation
  reaction anyway; H2 still shields CO regardless).

Reactions already parsed as shielded terms (the UCLCHEM parser does this)
are left in place; only their CSE geometry is (re)injected.

Column densities use the "plain n*r" approximation: the radial column from
the current radius outward is ``N_i = n_i(r) * r``. For :class:`CSEPhysics`
the radial extinction obeys ``Av = n * r / 1.87e21`` exactly, so the rate
term recovers ``r = column_scale / Av`` without the radius being plumbed
through the ODE right-hand side (see ``_shielding_length_cm`` in
``reactions.py``).

CO offers two self-shielding treatments, selected by
``config.co_shielding_method``:

- ``"vdb"`` (default): van Dishoeck & Black (1988) 2D ``N(CO) x N(H2)``
  table, as used by UCLCHEM.
- ``"oneband"``: Morris & Jura (1983) one-band analytic approximation for a
  constant-velocity outflow -- the classic circumstellar treatment. Needs
  an outflow velocity, so it is only applied to CSE runs.
"""

import math

from .physics import CSEPhysics
from .reactions import CIonizationReaction, COPhotoDissReaction, H2PhotoDissReaction

# N_H / Av [cm^-2 mag^-1] -- must match CSEPhysics.get_conditions.
_NH_PER_AV = 1.87e21

CO_SHIELDING_METHODS = ("vdb", "oneband")


def _cse_geometry(physics: CSEPhysics):
    """(column_scale [cm mag], velocity [cm/s]) for a CSE outflow.

    ``r = column_scale / Av`` inverts ``Av = n r / 1.87e21`` with
    ``n = mdot / (4 pi r^2 v mu m_H)``.
    """
    mdot_cgs = physics.mdot * physics.MSUN_G / physics.YR_S
    v_cgs = physics.vexp * physics.KM_CM
    column_scale = mdot_cgs / (4.0 * math.pi * v_cgs * physics.MU * physics.MH * _NH_PER_AV)
    return column_scale, v_cgs


# Generic dust-only photo reaction classes that a shielded term can replace.
_GENERIC_PHOTO = {"UMISTPhotoReaction", "UCLCHEMPhotonReaction"}
# Already-specialised shielded classes (e.g. from the UCLCHEM parser) -- left
# in place, only their CSE geometry is (re)injected.
_SHIELDED_PHOTO = {"COPhotoDissReaction", "CIonizationReaction", "H2PhotoDissReaction"}


def _match(reaction, reactants, products) -> bool:
    return set(reaction.reactants) == set(reactants) and set(reaction.products) == set(products)


def configure_self_shielding(network, config):
    """
    Rewrite CO / C / H2 photo reactions in ``network`` to self-shielded rate
    terms, in place. Returns ``network``.

    No-op when ``config.self_shielding`` is False. Safe to call on networks
    that have no CO/C/H2 or no photo reactions.
    """
    if not getattr(config, "self_shielding", True):
        return network

    species = {s.name for s in network.species}
    index = {s.name: i for i, s in enumerate(network.species)}

    physics = getattr(config, "physics_model", None)
    if isinstance(physics, CSEPhysics):
        column_scale, velocity_cms = _cse_geometry(physics)
        cloud_radius_pc = 1.0  # unused when column_scale > 0
    else:
        column_scale, velocity_cms = 0.0, 0.0
        cloud_radius_pc = getattr(
            physics, "cloud_radius_pc", getattr(config, "cloud_radius_pc", 1.0)
        )

    method = getattr(config, "co_shielding_method", "auto")
    is_cse = column_scale > 0.0
    if method == "auto":
        method = "oneband" if is_cse else "vdb"
    if method not in CO_SHIELDING_METHODS:
        raise ValueError(
            f"co_shielding_method must be 'auto' or one of {CO_SHIELDING_METHODS}, "
            f"got {method!r}"
        )
    if method == "oneband" and not is_cse:
        raise ValueError(
            "co_shielding_method='oneband' needs an outflow velocity; it only "
            "applies to CSE runs (config.physics_model must be a CSEPhysics)."
        )

    shield_c = getattr(config, "shield_c_ionization", False)
    shield_h2 = getattr(config, "shield_h2", False)

    def _configure(rx):
        """Inject CSE geometry into an (already specialised) shielded term."""
        rx.column_scale = column_scale
        rx.cloud_radius_pc = cloud_radius_pc
        if rx.__class__.__name__ == "COPhotoDissReaction":
            rx.velocity_cms = velocity_cms
            rx.co_shielding_method = method
        return rx

    def _carry_metadata(old, new):
        new.reaction_id = getattr(old, "reaction_id", None)
        new.uncertainty_flag = getattr(old, "uncertainty_flag", None)
        new.uncertainty_factor = getattr(old, "uncertainty_factor", 1.0)
        return _configure(new)

    new_reactions = []
    n_shielded = 0
    for reaction in network.reactions:
        cls = reaction.__class__.__name__

        # Already a shielded term (e.g. from the UCLCHEM parser): keep it,
        # just make sure its CSE geometry is current.
        if cls in _SHIELDED_PHOTO:
            new_reactions.append(_configure(reaction))
            continue

        if cls not in _GENERIC_PHOTO:
            new_reactions.append(reaction)
            continue

        replacement = None

        # CO + PHOTON -> O + C   (always, when self_shielding is on)
        if _match(reaction, ["CO"], ["O", "C"]) and {"CO", "H2"} <= species:
            replacement = COPhotoDissReaction(
                "COPHOTODISS", ["CO"], ["O", "C"],
                h2_species_index=index["H2"], co_species_index=index["CO"],
            )
            _carry_metadata(reaction, replacement)
            replacement.base_rate = float(getattr(reaction, "alpha", 2.0e-10))

        # C + PHOTON -> C+ + e-   (opt-in: config.shield_c_ionization)
        elif (
            shield_c
            and _match(reaction, ["C"], ["C+", "e-"])
            and {"C", "H2"} <= species
        ):
            replacement = CIonizationReaction(
                "CPHOTOION", ["C"], ["C+", "e-"],
                alpha=float(getattr(reaction, "alpha", 3.5e-10)),
                gamma=float(getattr(reaction, "gamma", 3.0)),
                c_species_index=index["C"], h2_species_index=index["H2"],
            )
            _carry_metadata(reaction, replacement)

        # H2 + PHOTON -> H + H   (opt-in: config.shield_h2)
        elif shield_h2 and _match(reaction, ["H2"], ["H", "H"]) and "H2" in species:
            replacement = H2PhotoDissReaction(
                "H2PHOTODISS", ["H2"], ["H", "H"], h2_species_index=index["H2"],
            )
            _carry_metadata(reaction, replacement)

        if replacement is not None:
            new_reactions.append(replacement)
            n_shielded += 1
        else:
            new_reactions.append(reaction)

    network.reactions = new_reactions
    network._n_self_shielded = n_shielded
    network._co_shielding_method = method  # resolved (never "auto")
    return network
