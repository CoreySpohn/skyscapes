"""Band integration of scalar-wavelength functions with optional spectral knots."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.spectral import Band, bin_mean
from jaxtyping import Array


class Bandpass(eqx.Module):
    """Tabulated photon-counting response with linear interpolation.

    Construct from concrete arrays; the resulting module can be passed to
    JIT, vmap and grad. Wavelengths are finite, positive and strictly
    increasing; transmission is finite, nonnegative and not identically
    zero. Transmission is zero outside the table, including for a top hat.

    Attributes:
        wavelength_nm: Sorted wavelength knots [nm], shape ``(N,)``, N >= 2.
        transmission: Photon-counting response at each knot, shape ``(N,)``.
        band: Concrete wavelength-domain geometry in nm.
    """

    wavelength_nm: Array
    transmission: Array

    band: Band = eqx.field(static=True, init=False)

    def __post_init__(self):
        """Reject tables without a finite, positive-area interpolation domain."""
        self.wavelength_nm = jnp.asarray(self.wavelength_nm)
        self.transmission = jnp.asarray(self.transmission)
        wave, response = self.wavelength_nm, self.transmission
        if wave.ndim != 1 or wave.size < 2 or response.shape != wave.shape:
            raise ValueError("Bandpass tables must share shape (N,) with N >= 2.")
        wave, response = np.asarray(wave), np.asarray(response)
        if (
            not np.all(np.isfinite(wave))
            or np.any(wave <= 0)
            or np.any(np.diff(wave) <= 0)
        ):
            raise ValueError(
                "Bandpass wavelengths must be finite, positive and sorted."
            )
        if (
            not np.all(np.isfinite(response))
            or np.any(response < 0)
            or not np.any(response > 0)
        ):
            raise ValueError(
                "Bandpass transmission must be finite, nonnegative and nonzero."
            )

        self.band = Band(wave[0], wave[-1])

    @classmethod
    def tophat(cls, center_nm: float, width_nm: float) -> "Bandpass":
        """Unit transmission across a positive full width [nm] about center_nm."""
        return cls(
            jnp.asarray([center_nm - width_nm / 2, center_nm + width_nm / 2]),
            jnp.ones(2),
        )

    def __call__(self, wavelength_nm: Array) -> Array:
        """Linearly interpolate the response, returning zero outside the table."""
        return jnp.interp(
            wavelength_nm, self.wavelength_nm, self.transmission, left=0.0, right=0.0
        )

    @property
    def low_nm(self) -> Array:
        """Lower edge of the tabulated wavelength domain [nm]."""
        return self.wavelength_nm[0]

    @property
    def high_nm(self) -> Array:
        """Upper edge of the tabulated wavelength domain [nm]."""
        return self.wavelength_nm[-1]

    @property
    def mean_nm(self) -> Array:
        """Mean photon wavelength, integral(lambda*S)/integral(S), in nm.

        This source-independent mean weights wavelengths by the photonic
        response S (Bessell and Murphy 2012, PASP 124, 140, Appendix A).
        It is the detected mean wavelength for a flat incident photon flux
        density. Two-node quadrature integrates each linear response
        segment's area and quadratic first moment exactly.
        """
        centers = jnp.atleast_1d(self.band.center_nm)
        widths = jnp.atleast_1d(self.band.width_nm)
        return bin_mean(
            lambda w: w,
            centers,
            widths,
            n_nodes=2,
            knots_nm=self.wavelength_nm,
            max_knots=len(self.wavelength_nm) + 1,
            weight_fn=self,
        )[0]
