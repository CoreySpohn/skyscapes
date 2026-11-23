"""Knot digests and band-rule lookups for tabulated source spectra.

A tabulated spectrum keeps a static digest of its knots, computed once on the
host, so the operator that integrates it on a band rule can be found inside a
traced program (``optixstuff.spectral.Resolution``). A planet whose spectrum is
tabulated is integrated as the product of its own table and the host star's,
on the merged knots of the two; :class:`ProductTable` names that product to
``Resolution.build``, and :func:`product_rule` finds its operator again from
the two static digests.
"""

from __future__ import annotations

import jax
import numpy as np
from optixstuff.spectral import SpectralTable, digest_knots

SMOOTH_DEGREE = 3
"""Degree declared for a smooth, non-polynomial factor between two knots.

The operator places ``ceil((order + degree) / 2)`` Gauss points on every cell
between knots. A smooth factor (a Planck spectrum, a phase-free gray albedo) is
not a polynomial; counting it as a cubic puts enough points on cells as narrow
as a tabulated grid's for its error to sit at round-off.
"""

_TRACED = (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError)


def table_digest(knots, method: str) -> str | None:
    """Return the knot digest of concrete knots, or None.

    None marks a table that cannot be digested: knots that are traced (the
    table was constructed under a trace) or that do not form a strictly
    increasing grid of at least two wavelengths (no operator can be built on
    them). ``knots`` is an array or a callable that returns one, so a model
    whose knots are read through a method can be digested under a trace.

    Args:
        knots: Knot wavelengths [nm], or a callable returning them.
        method: Name of the interpolation between the knots.

    Returns:
        The ``"method:hash"`` digest, or None.
    """
    try:
        knots = knots() if callable(knots) else knots
        if isinstance(knots, jax.core.Tracer):
            return None
        return digest_knots(knots, method)
    except _TRACED:
        return None
    except ValueError:
        return None


def is_table(source) -> bool:
    """Whether ``source`` is a tabulated spectrum with a knot digest attribute.

    Args:
        source: A star, a local-zodi model or a physical model.

    Returns:
        True when ``source`` implements ``SpectralTable``.
    """
    return isinstance(source, SpectralTable)


def check_digest(table) -> None:
    """Refuse a table whose digest is missing or no longer matches its knots.

    Args:
        table: A ``SpectralTable``.
    """
    name = type(table).__name__
    digest = table.knot_digest
    if digest is None:
        raise ValueError(
            f"{name} has no knot digest: it was constructed under a trace or on "
            "knots that are not a strictly increasing grid of at least two "
            "wavelengths; construct the table on the host"
        )
    method = digest.rpartition(":")[0]
    if digest_knots(table.spectral_breaks_nm(), method) != digest:
        raise ValueError(
            f"the knot digest of {name} does not match its knots: a knot grid "
            "was replaced after construction; rebuild the table"
        )


def _star_part(star) -> tuple[str, int]:
    """Digest and degree that name the star's factor in a product."""
    if not is_table(star):
        return "smooth", SMOOTH_DEGREE
    if star.knot_digest is None:
        raise LookupError(
            f"{type(star).__name__} has no knot digest (it was constructed under "
            "a trace); construct the star on the host"
        )
    return star.knot_digest, star.spectral_degree


def product_method(model_digest: str, star_digest: str, degree: int) -> str:
    """Name a planet table times its host star as an interpolation method.

    The name holds both static digests, so it fixes the merged knots without
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
    """A tabulated planet spectrum times its host star, on the merged knots.

    Host only: built by ``Scene.spectral_tables`` and read by
    ``Resolution.build``. Its knots are the planet's and the star's (a smooth
    star adds none), restricted to the wavelengths both tables cover, and its
    degree is the sum of theirs. It holds no interpolant, so the operator is the
    ``"fine"`` kind: the planet flux is evaluated on the operator's fine grid and
    reduced, or the ``"identity"`` kind when every merged knot is a piece edge.

    Attributes:
        knot_digest: Digest of the merged knots under :func:`product_method`.
        spectral_degree: Declared degree of the product between knots.
    """

    knot_digest: str
    spectral_degree: int

    def __init__(self, model, star):
        """Merge the knots of a tabulated physical model and its host star.

        Args:
            model: A physical model that declares knots.
            star: The host star.
        """
        check_digest(model)
        star_digest, star_degree = _star_part(star)
        knots = np.asarray(model.spectral_breaks_nm(), dtype=float)
        if is_table(star):
            check_digest(star)
            star_knots = np.asarray(star.spectral_breaks_nm(), dtype=float)
            low = max(knots[0], star_knots[0])
            high = min(knots[-1], star_knots[-1])
            if low >= high:
                raise ValueError(
                    f"the tables of {type(model).__name__} and "
                    f"{type(star).__name__} share no wavelengths"
                )
            knots = np.union1d(knots, star_knots)
            knots = knots[(knots >= low) & (knots <= high)]
        self.spectral_degree = model.spectral_degree + star_degree
        self._knots = knots
        method = product_method(model.knot_digest, star_digest, self.spectral_degree)
        self.knot_digest = digest_knots(knots, method)

    def spectral_breaks_nm(self) -> np.ndarray:
        """Return the merged knots [nm].

        Returns:
            Strictly increasing knot wavelengths in nm.
        """
        return self._knots


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
    if model.knot_digest is None:
        raise LookupError(
            f"{name} has no knot digest: it was constructed under a trace or on "
            "fewer than two increasing knots; construct the model on the host"
        )
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
