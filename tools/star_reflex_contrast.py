"""Measure how the star reflex motion changes phase-mode planet contrasts.

Loads the demo ExoVista scene twice with ``planet_spectra="phase"``: once as
written, and once with the star's state columns zeroed so the barycentric
planet states are read as star-relative. Prints, per planet, the median,
99th-percentile and maximum relative difference of ``System.contrasts`` at one
wavelength over every FITS epoch.

Run from the repository root:

    python tools/star_reflex_contrast.py [wavelength_nm]
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from hwoutils.constants import J2000_JD, d2s, yr2s
from jax import config

config.update("jax_enable_x64", True)

from skyscapes.datasets import fetch_scene  # noqa: E402
from skyscapes.io import from_exovista  # noqa: E402
from skyscapes.io.exovista import _star_extension  # noqa: E402


def main(wavelength_nm=600.0):
    """Print the contrast change per planet for the demo scene."""
    source = fetch_scene()
    with tempfile.TemporaryDirectory() as tmp:
        zeroed = Path(tmp) / "star_rows_zeroed.fits"
        shutil.copy(source, zeroed)
        with fits.open(zeroed, mode="update") as hdul:
            star = hdul[_star_extension(hdul)].data
            offset = np.linalg.norm(star[0, 9:12].astype(float))
            star[:, 9:15] = 0.0
        subtracted = from_exovista(source, load_disk=False, planet_spectra="phase")
        barycentric = from_exovista(zeroed, load_disk=False, planet_spectra="phase")
    with fits.open(source) as hdul:
        elapsed = hdul[_star_extension(hdul)].data[:, 0].astype(float)
    times = J2000_JD + elapsed * (yr2s / d2s)
    new = np.asarray(subtracted.contrasts(wavelength_nm, times))
    old = np.asarray(barycentric.contrasts(wavelength_nm, times))
    rel = np.abs(new - old) / np.maximum(np.abs(old), 1e-30)
    print(f"star barycentric offset [AU]: {offset:.7f}; epochs: {len(times)}")
    print(f"relative contrast change at {wavelength_nm:g} nm [percent]")
    print("planet   median      p99      max")
    for k, row in enumerate(rel):
        med, p99, top = 100 * np.array(
            [np.median(row), np.percentile(row, 99), row.max()]
        )
        print(f"{k:6d} {med:8.4f} {p99:8.4f} {top:8.4f}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 600.0)
