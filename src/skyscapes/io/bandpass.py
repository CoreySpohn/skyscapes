"""Local two-column photon-counting bandpass tables."""

from pathlib import Path

import numpy as np
from optixstuff.spectral import Transmission


def bandpass_from_table(path, *, wavelength_unit="nm") -> Transmission:
    """Read wavelength and transmission columns from a local text file.

    Accept whitespace or comma separators and ``#`` comments. Wavelength
    units are ``"nm"``, ``"um"`` or ``"angstrom"``; transmission is a
    nonnegative photon-counting response. Rows must have increasing
    wavelengths. No data are fetched from the network.
    """
    factors = {"nm": 1.0, "um": 1000.0, "angstrom": 0.1}
    if wavelength_unit not in factors:
        raise ValueError("wavelength_unit must be 'nm', 'um' or 'angstrom'.")
    with Path(path).open() as stream:
        table = np.loadtxt((line.replace(",", " ") for line in stream), ndmin=2)
    if table.shape[1] != 2:
        raise ValueError("Transmission table must contain exactly two columns.")
    return Transmission(table[:, 0] * factors[wavelength_unit], table[:, 1])
