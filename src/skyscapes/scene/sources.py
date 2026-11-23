"""Every source of a scene on the nodes of one band rule."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .star import _NodeStar


class Sources(eqx.Module):
    """The scene's sources as node-effective values on one band rule.

    Built by ``Scene.sources``, never by hand. Each value is the number at a
    node whose weighted sum with a response at the nodes is the band integral
    of the source times that response: the star and a tabulated planet are
    reduced from their tables, so the sum is exact in the source however
    rough its spectrum, and exact in the response to the degree of the rule.
    Contract the node axes with the quadrature that built them.

    Attributes:
        unscaled_star: Node-effective photon flux density of the host star
            without its ``flux_scale`` [ph s^-1 m^-2 nm^-1], shape ``(W, Q, T)``.
        star_scale: The host star's ``flux_scale``.
        planets: Node-effective photon flux density of each planet, times its
            own and (for reflected light) the star's scale
            [ph s^-1 m^-2 nm^-1], shape ``(W, Q, K, T)``; zero on padded slots.
        local_zodi: Node-effective local-zodi photon radiance, times its scale
            [ph s^-1 m^-2 nm^-1 sr^-1], shape ``(W, Q, T)``, or None for a scene
            without one.
        star_diameter_arcsec: Apparent diameter of the host star [arcsec].
        star_luminosity_Lsun: Luminosity of the host star [solar luminosities],
            or None when the star does not carry one.
    """

    unscaled_star: Array
    star_scale: Array
    planets: Array
    local_zodi: Array | None
    star_diameter_arcsec: Array
    star_luminosity_Lsun: Array | None

    @property
    def star(self) -> Array:
        """Node-effective star flux density with its scale, shape ``(W, Q, T)``."""
        scale = jnp.asarray(self.star_scale)
        return scale[..., None, None, None] * self.unscaled_star

    @property
    def contrast(self) -> Array:
        """Star-weighted node contrast, ``planets / star``, shape ``(W, Q, K, T)``.

        A transport quantity: multiplied by the node-effective star it returns
        the planet's node-effective flux, which is what a route that takes a
        contrast needs. It is not a physical contrast to plot or report. Where
        the star's node value is near zero it is large while the product stays
        right, and on a tabulated planet it differs from the point contrast
        by the planet's own spectral structure.
        """
        return self.planets / self.star[..., None, :]

    def node_star(self, w=None, q=None, t=None) -> _NodeStar:
        """Return the host star as its node-effective values, for a disk read.

        With ``w`` and ``q`` it is the star at one node, which a route that
        renders node by node passes to ``Scene.photon_radiance(..., star=)``.
        Without them it holds every node in order, for a read vectorised over
        the nodes (``Scene.photon_radiance_at(quadrature.nodes_nm, ...)``).
        A disk defined relative to the star then carries the star's band
        integral rather than its point sample. Unbatched records only.

        Args:
            w: Band index, or None for every band.
            q: Node index within the band, or None for every node.
            t: Time index, or None when the record holds one time.

        Returns:
            A star whose photon flux density is the node-effective value.
        """
        values = self.unscaled_star
        if t is None:
            if values.shape[-1] != 1:
                raise ValueError(
                    "these sources hold several times; pass the time index t"
                )
            t = 0
        index = (
            slice(None) if w is None else w,
            slice(None) if q is None else q,
            t,
        )
        return _NodeStar(
            values=values[index],
            diameter_arcsec=self.star_diameter_arcsec,
            luminosity_Lsun=self.star_luminosity_Lsun,
            flux_scale=self.star_scale,
        )
