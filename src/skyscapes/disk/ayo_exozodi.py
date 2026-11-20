"""The AYO analytic exozodi as a disk model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_au
from jaxtyping import Array
from zodi.jax import AYOExozodi

from .base import AbstractDisk


class AYOExozodiDisk(AbstractDisk):
    """Analytic AYO exozodi, expressed as a scene disk.

    Wraps :class:`zodi.jax.AYOExozodi`: the radiance depends only on the
    projected physical separation ``dist_pc * hypot(east, north)`` from the
    star, so the disk is face-on symmetric and ignores time, the star and the
    scene midplane angles. It diverges at zero separation, so sample it away
    from the star pixel.

    Attributes:
        model: The scalar AYO exozodi (level in zodis and one-zodi surface
            brightness).
    """

    model: AYOExozodi = AYOExozodi()

    def spectral_breaks_nm(self) -> np.ndarray:
        """Wavelength breaks of the wrapped model (none)."""
        return self.model.spectral_breaks_nm()

    def photon_radiance_at(
        self, wavelength_nm, time_d, east_arcsec, north_arcsec, *, star, dist_pc
    ) -> Array:
        """Photons/s/m^2/nm/sr, wavelength axes first, then the sky shape."""
        wave = jnp.asarray(wavelength_nm)
        separation = jnp.hypot(east_arcsec, north_arcsec)
        wave = wave.reshape(wave.shape + (1,) * jnp.ndim(separation))
        return self.model.photon_radiance(
            wave, separation_au=arcsec_to_au(separation, dist_pc)
        )
