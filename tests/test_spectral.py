"""Local response-table loading checks."""

import numpy as np
import pytest
from hwoutils.spectral import Transmission

from skyscapes.io import bandpass_from_table


@pytest.mark.parametrize(
    "unit, scale, delimiter",
    [("nm", 1.0, " "), ("um", 1000.0, ","), ("angstrom", 0.1, " ")],
)
def test_bandpass_table_units(tmp_path, unit, scale, delimiter):
    """Table values round-trip to rtol 2e-15 for unit-conversion rounding.

    The owner receives the triangle vertices (400,0), (500,1), (700,0).
    """
    path = tmp_path / "band.txt"
    np.savetxt(
        path,
        np.array([[400 / scale, 0], [500 / scale, 1], [700 / scale, 0]]),
        delimiter=delimiter,
        header="wavelength, transmission",
    )
    band = bandpass_from_table(path, wavelength_unit=unit)
    assert isinstance(band, Transmission)
    np.testing.assert_allclose(band.wavelength_nm, [400, 500, 700], rtol=2e-15, atol=0)
    np.testing.assert_array_equal(band.transmission, [0, 1, 0])
