"""Static knots and band-rule lookups for tabulated source spectra.

A tabulated spectrum holds its knots as a static ``optixstuff.spectral.Knots``
(immutable float64 host data, hashed by its digest), never as a leaf, so the
operator that integrates it on a band rule can be found inside a traced
program (``optixstuff.spectral.Resolution``) and no fit can move a knot. A
planet whose spectrum is tabulated is integrated as the product of its own
table and the host star's, on the union of the two sets of knots;
:class:`ProductTable` names that product to ``Resolution.build``, and
:func:`product_rule` finds its operator again from the two static digests.
"""

from __future__ import annotations

import functools

import numpy as np
from hwoutils.conversions import wavenumber_cm_to_nm
from hwoutils.leaves import HostGrid
from optixstuff.spectral import (
    Knots,
    ProductEdges,
    SpectralTable,
    digest_knots,
    same_wavelength,
)

SMOOTH_DEGREE = 3
"""Degree declared for a smooth, non-polynomial factor between two knots.

The operator places ``ceil((order + degree) / 2)`` Gauss points on every cell
between knots. A smooth factor (a Planck spectrum, a phase-free gray albedo) is
not a polynomial; counting it as a cubic puts enough points on cells as narrow
as a tabulated grid's for its error to sit at round-off.
"""


def as_host_grid(values) -> HostGrid:
    """Return a ``HostGrid`` unchanged, or build one from concrete values.

    Args:
        values: A ``HostGrid`` or a concrete 1-D array-like.

    Returns:
        The ``HostGrid``.
    """
    return values if isinstance(values, HostGrid) else HostGrid(values)


# Keyed by the grid's digest; an entry is one knot array, about 14 MB per grid at
# R = 1e6, so 32 entries hold about 450 MB in the extreme case.
@functools.lru_cache(maxsize=32)
def knots_of(grid: HostGrid) -> Knots:
    """Knots of a static host grid in nm, cached on its digest.

    Args:
        grid: A wavelength grid [nm].

    Returns:
        The grid as ``Knots``; raises when it is not a valid knot grid.
    """
    return Knots(grid.values)


@functools.lru_cache(maxsize=32)
def wavenumber_knots(nu_grid: HostGrid) -> Knots:
    """Knots [nm] of a static wavenumber grid [cm^-1], sorted, cached.

    The wavenumber grid stays the exact float64 values the opacity engine was
    built on; the knots are its float64 conversion to nm, never the other way
    round (a round trip through nm is not bitwise ``nu``).

    Args:
        nu_grid: Wavenumber grid [cm^-1].

    Returns:
        The grid wavelengths [nm] as ``Knots``.
    """
    return Knots(np.sort(np.asarray(wavenumber_cm_to_nm(nu_grid.values))))


def is_table(source) -> bool:
    """Whether ``source`` is a tabulated spectrum with static knots.

    Args:
        source: A star, a local-zodi model or a physical model.

    Returns:
        True when ``source`` implements ``SpectralTable`` and has knots.
    """
    return (
        isinstance(source, SpectralTable) and getattr(source, "knots", None) is not None
    )


def _star_part(star) -> tuple[str, int]:
    """Digest and degree that name the star's factor in a product."""
    if not is_table(star):
        return "smooth", SMOOTH_DEGREE
    return star.knot_digest, star.spectral_degree


def product_method(model_digest: str, star_digest: str, degree: int) -> str:
    """Name a planet table times its host star as an interpolation method.

    The name holds both static digests, so it fixes the cell edges without
    them, and a rule built for the product can be found under a trace.

    Args:
        model_digest: Knot digest of the planet's physical model.
        star_digest: Knot digest of the star, or ``"smooth"``.
        degree: Declared polynomial degree of the product between knots.

    Returns:
        A method name without a colon, for ``digest_knots``.
    """
    model = model_digest.replace(":", "=")
    star = star_digest.replace(":", "=")
    return f"product {model} {star} degree {degree}"


class ProductTable:
    """A tabulated planet spectrum times its host star, on the union of knots.

    Host only: built by ``Scene.spectral_tables`` and read by
    ``Resolution.build``. Its cell edges are the raw union of the planet's and
    the star's knots, restricted to the wavelengths both tables cover, with no
    tolerance merge: a star knot within the wavelength tolerance of a planet
    knot stays a distinct edge, so the product of the two interpolants is
    integrated exactly on every cell, and the operator applies the
    ``Reduction.build`` edge rule to it as to a single table. A smooth star
    adds no knots. The degree is the sum of the two. It holds no interpolant,
    so the operator is the ``"fine"`` kind: the planet flux is evaluated on the
    operator's fine grid and reduced, or the ``"identity"`` kind when every
    edge is a piece edge.

    Attributes:
        knots: The cell edges, static ``ProductEdges`` (not ``Knots``: edges
            closer than the tolerance are allowed).
        knot_digest: Digest of the component tables under
            :func:`product_method`, which names both component digests.
        spectral_degree: Declared degree of the product between knots.
    """

    knots: ProductEdges
    knot_digest: str
    spectral_degree: int

    def __init__(self, model, star):
        """Take the union of the knots of a tabulated model and its host star.

        Args:
            model: A physical model that declares knots.
            star: The host star.
        """
        star_digest, star_degree = _star_part(star)
        edges = model.knots.nm
        if is_table(star):
            low = max(edges[0], star.knots.nm[0])
            high = min(edges[-1], star.knots.nm[-1])
            if low >= high or same_wavelength(low, high):
                raise ValueError(
                    f"the tables of {type(model).__name__} and "
                    f"{type(star).__name__} share no wavelengths"
                )
            merged = np.unique(np.concatenate([edges, star.knots.nm]))
            keep = ((merged >= low) | same_wavelength(merged, low)) & (
                (merged <= high) | same_wavelength(merged, high)
            )
            edges = merged[keep]
        self.spectral_degree = model.spectral_degree + star_degree
        self.knots = ProductEdges(edges)
        method = product_method(model.knot_digest, star_digest, self.spectral_degree)
        self.knot_digest = digest_knots(self.knots, method)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the cell edges [nm].

        Returns:
            Strictly increasing wavelengths in nm.
        """
        return self.knots.values


def product_rule(resolution, quadrature, model, star):
    """Return the operator of a planet table times its star, under a trace.

    Found from the two static digests: the rule whose knot digest carries
    :func:`product_method` of them, on this quadrature's layout.

    Args:
        resolution: The ``Resolution`` built for the scene's tables.
        quadrature: The band rule whose nodes the values are wanted on.
        model: A physical model that declares knots.
        star: The host star.

    Returns:
        The ``Reduction`` built for the product on this quadrature.
    """
    name = type(model).__name__
    layout = quadrature.layout_digest
    if layout is None:
        raise LookupError(
            "a hand-made quadrature has no piece layout and no rule; build it "
            "with Quadrature.build"
        )
    star_digest, star_degree = _star_part(star)
    method = product_method(
        model.knot_digest, star_digest, model.spectral_degree + star_degree
    )
    for (rule_layout, digest), rule in zip(
        resolution.keys, resolution.rules, strict=True
    ):
        if rule_layout == layout and digest.rpartition(":")[0] == method:
            return rule
    raise LookupError(
        f"no rule for {name} under {type(star).__name__} on this quadrature: "
        "build it with Resolution.build(quadrature, *scene.spectral_tables()). "
        "A band slice has no rule of its own; carry the rows of the full rule's "
        "values"
    )
