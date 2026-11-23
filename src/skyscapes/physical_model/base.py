"""Abstract physical-model interface for a planet's radiative output.

Subclasses own whatever spectral-physics parameters describe "this kind of
planet" (Lambertian, grid-interpolated, radiative-transfer, thermal
emission). ``photon_flux_density`` returns photon flux at the observer.
Its default multiplies contrast by the host star's photon flux; thermal
models provide flux directly. The phase-dependent hook maps
(phase_angle, star-planet distance, wavelength, planet radius) onto a
contrast (flux ratio relative to the host star).
Time-dependent models override ``contrast_at_time`` instead of identifying
physical epochs with orbital phase.

The planet radius is supplied as a method argument rather than a class
field -- ``Rp_Rearth`` is a property of the planet, not the physical
model. ``Scene`` forwards ``planet.Rp_Rearth`` to the physical
model at call time.

``node_photon_flux_density`` reads a planet on the nodes of a band rule. A
model with no knots returns its contrast at the nodes times the host star's
node-effective flux (reflected light) or its own flux at the nodes (thermal
light). A model that declares knots is integrated as a table: its flux is
evaluated on the fine grid of the operator built for its knots and the star's,
and reduced.
"""

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._tables import product_rule, table_digest


class AbstractPhysicalModel(eqx.Module):
    """Spectral-physics layer for a planet.

    Concrete subclasses own per-planet physical-model parameters (albedo,
    grid contrast cube, ExoJax composition, etc.). The host planet's
    radius is supplied at call time via ``contrast(..., Rp_Rearth=...)``.

    A model that overrides :meth:`spectral_breaks_nm` declares knots: it is a
    tabulated spectrum. Its ``knot_digest`` is computed once at construction
    from those knots and the class's interpolation (``spectral_degree``
    between knots), so the operator that integrates it can be found inside a
    traced program; it is None for a model constructed under a trace and for
    a model without knots. A composite reports the union of its parts.

    Attributes:
        knot_digest: Static digest of the declared knots and the interpolation
            method, or None.
        spectral_degree: Polynomial degree of the spectrum between knots.
    """

    knot_digest: str | None = eqx.field(
        static=True, init=False, default=None, repr=False
    )
    spectral_degree: ClassVar[int] = 1
    _knot_method: ClassVar[str] = "linear"

    def __post_init__(self):
        """Digest the declared knots on the host; None under a trace."""
        if self._tabulated:
            self.knot_digest = table_digest(self.spectral_breaks_nm, self._knot_method)

    @property
    def _tabulated(self) -> bool:
        """Whether the model declares knots (overrides ``spectral_breaks_nm``)."""
        return (
            type(self).spectral_breaks_nm
            is not AbstractPhysicalModel.spectral_breaks_nm
        )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Analytic spectra declare no integration breaks; tabulated ones override.

        Returns:
            Empty 1-D array of wavelength breaks [nm]; tabulated models override it.
        """
        return np.empty(0)

    def photon_flux_density(
        self,
        wavelength_nm: Array,
        time_d: Array,
        *,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        Rp_Rearth: Array,
        star,
        dist_pc: Array,
    ) -> Array:
        """Planet photons [ph s^-1 m^-2 nm^-1], shape (K, T) + wavelength shape.

        Contrast and its spectral parameters belong to this model. Scene
        supplies orbix star-relative phase and distance, both (K, T), and
        Planet supplies radii (K,). Time is (T,) days from the scene epoch.
        Each wavelength returns stellar photon density times model contrast.
        Thermal kernels evaluate that product directly to retain dynamic range.
        No integration or band width enters this method.

        Args:
            wavelength_nm: Wavelengths [nm], any shape.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            phase_angle_rad: Star-planet-observer phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            star: Host ``AbstractStar``, providing the stellar photon flux.
            dist_pc: Distance of the system from the observer [pc].

        Returns:
            Photon flux density at the observer [ph s^-1 m^-2 nm^-1], shape
            ``(K, T) + wavelength_nm.shape``.
        """
        wave = jnp.asarray(wavelength_nm)

        def at_wavelength(w):
            return self._photon_flux_density(
                w,
                time_d=time_d,
                phase_angle_rad=phase_angle_rad,
                star_planet_distance_AU=star_planet_distance_AU,
                Rp_Rearth=Rp_Rearth,
                star=star,
                dist_pc=dist_pc,
            )

        values = jax.vmap(at_wavelength)(wave.reshape(-1))
        return jnp.moveaxis(values, 0, -1).reshape(phase_angle_rad.shape + wave.shape)

    def node_photon_flux_density(
        self,
        quadrature,
        resolution,
        time_d: Array,
        *,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        Rp_Rearth: Array,
        star,
        star_nodes: Array,
        dist_pc: Array,
    ) -> Array:
        """Node-effective planet photon flux density [ph s^-1 m^-2 nm^-1].

        The values on the band rule's nodes whose weighted sum with a response
        of degree below a piece's order is the band integral of the planet's
        flux times that response. A model without knots returns its contrast
        at the nodes times ``star_nodes``, exact to that degree for a smooth
        contrast under any star. A model that declares knots evaluates its
        flux on the fine grid of the operator ``resolution`` holds for its
        knots and the star's, and reduces it, exact in both tables. Thermal
        models override this method to return their flux at the nodes.

        Args:
            quadrature: Band rule from ``optixstuff.spectral.Quadrature.build``.
            resolution: ``optixstuff.spectral.Resolution`` built for the rule
                and ``Scene.spectral_tables()``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            phase_angle_rad: Star-planet-observer phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            star: Host ``AbstractStar``.
            star_nodes: The host star's node-effective photon flux density on
                ``quadrature``, times its ``flux_scale``, shape ``(W, Q, T)``.
            dist_pc: Distance of the system from the observer [pc].

        Returns:
            Node-effective photon flux density, shape ``(W, Q, K, T)``.
        """
        geometry = dict(
            phase_angle_rad=phase_angle_rad,
            star_planet_distance_AU=star_planet_distance_AU,
            Rp_Rearth=Rp_Rearth,
            star=star,
            dist_pc=dist_pc,
        )
        if self._tabulated:
            rule = product_rule(resolution, quadrature, self, star)
            points = quadrature.nodes_nm if rule.kind == "identity" else rule.fine_nm
            return rule.apply(self._flux_at(points, time_d, **geometry))
        nodes = quadrature.nodes_nm

        def at_node(wavelength_nm):
            return self.contrast_at_time(
                phase_angle_rad,
                star_planet_distance_AU,
                wavelength_nm,
                Rp_Rearth,
                time_d=time_d,
                star=star,
                dist_pc=dist_pc,
            )

        contrast = jax.vmap(at_node)(nodes.reshape(-1))
        contrast = contrast.reshape(nodes.shape + jnp.shape(phase_angle_rad))
        return contrast * star_nodes[:, :, None, :]

    def _flux_at(self, wavelength_nm, time_d, **geometry):
        """Photon flux with the wavelength axes first, shape wave + ``(K, T)``."""
        flux = self.photon_flux_density(wavelength_nm, time_d, **geometry)
        return jnp.moveaxis(flux, (0, 1), (-2, -1))

    def _photon_flux_density(
        self,
        wavelength_nm: Array,
        *,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        Rp_Rearth: Array,
        time_d: Array,
        star,
        dist_pc: Array,
    ) -> Array:
        """Planet photon flux density at the observer [ph/s/m^2/nm].

        The default multiplies ``contrast_at_time`` by stellar flux.
        Thermal models override this method to evaluate flux directly.

        Args:
            wavelength_nm: Scalar wavelength [nm].
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            dist_pc: Observer distance supplied by the system [pc].
            star: Host ``AbstractStar``, providing observer flux.

        Returns:
            Photon flux density, shape ``(K, T)``.
        """
        contrast = self.contrast_at_time(
            phase_angle_rad=phase_angle_rad,
            star_planet_distance_AU=star_planet_distance_AU,
            wavelength_nm=wavelength_nm,
            Rp_Rearth=Rp_Rearth,
            time_d=time_d,
            star=star,
            dist_pc=dist_pc,
        )
        return contrast * star.photon_flux_density(
            wavelength_nm, time_d, dist_pc=dist_pc
        )

    @abstractmethod
    def contrast(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
    ) -> Array:
        """Return planet-to-star contrast.

        Args:
            phase_angle_rad: Star-planet-observer phase angle ``beta``,
                shape ``(K, T)`` for K planets and T times.
            star_planet_distance_AU: Star-planet distance, shape ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm]. Use ``Scene.contrast_cube``
                to evaluate an array of wavelengths.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``,
                supplied by the host ``Planet``.

        Returns:
            Flux-ratio contrast, shape ``(K, T)``.
        """

    def contrast_at_time(
        self,
        phase_angle_rad: Array,
        star_planet_distance_AU: Array,
        wavelength_nm: Array,
        Rp_Rearth: Array,
        *,
        time_d: Array,
        star=None,
        dist_pc: Array | None = None,
    ) -> Array:
        """Return contrast with the observation epochs and the host star available.

        The default forwards the four radiative arguments to ``contrast``
        and ignores ``time_d`` and ``star``. Existing phase-dependent models
        therefore keep their behavior without implementing another method.
        A model whose planet flux does not scale with the star's (thermal
        emission) overrides this method and divides by the star's flux.

        Args:
            phase_angle_rad: Phase angle [rad], shape ``(K, T)``.
            star_planet_distance_AU: Star-planet distance [AU], shape ``(K, T)``.
            wavelength_nm: Scalar wavelength [nm], as accepted by ``contrast``.
            Rp_Rearth: Planet radius [Earth radii], shape ``(K,)``.
            time_d: Observation times in days from the scene epoch, shape ``(T,)``.
            star: The host ``AbstractStar``; ``Scene`` always supplies it.
            dist_pc: Observer distance supplied by the system [pc].

        Returns:
            Planet-to-star flux ratio, shape ``(K, T)``.
        """
        return self.contrast(
            phase_angle_rad, star_planet_distance_AU, wavelength_nm, Rp_Rearth
        )
