"""Back-compat shim re-exporting the PSG cross-section O3 setup.

The original implementation lived here; it has since been generalized
to :mod:`skyscapes.physical_model.exojax.psg_xs`. This module keeps the
``O3ChappuisOpacity`` name so existing imports continue to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Array

from .psg_xs import (
    PsgCrossSectionOpacity,
    _load_psg_xs_table,
    fetch_psg_xs,
)

PSG_O3_URL = "https://psg.gsfc.nasa.gov/data/linelists/xuv/data/o3.txt"

# O3 molar mass [g/mol] -- matches the value HITRAN reports via
# ``radis``'s ``molmass_hitran`` for isotope 1 (the principal 16O3).
O3_MOLMASS = 47.99820


def fetch_psg_o3(cache_dir: str | Path | None = None) -> Path:
    """Download the PSG O3 cross-section file once and cache it locally.

    Thin wrapper around :func:`fetch_psg_xs` that supplies the O3 URL
    and a stable cache filename.
    """
    return fetch_psg_xs(PSG_O3_URL, "psg_o3_chappuis.txt", cache_dir)


class O3ChappuisOpacity(PsgCrossSectionOpacity):
    """PSG cross-section opacity for O3 (Serdyuchenko et al. 2014)."""

    def __init__(
        self,
        nu_grid: Array | np.ndarray,
        cache_dir: str | Path | None = None,
        xs_table_path: str | Path | None = None,
    ):
        """Construct an O3 PSG cross-section adapter.

        Args:
            nu_grid: Wavenumber grid [cm^-1], shape ``(n_nu,)``.
            cache_dir: Directory for the cached PSG file. Defaults to
                ``~/.cache/skyscapes/``.
            xs_table_path: Optional explicit path to a PSG-format file
                (useful in tests to avoid a network fetch).
        """
        super().__init__(
            name="O3",
            url=PSG_O3_URL,
            molmass=O3_MOLMASS,
            nu_grid=nu_grid,
            cache_dir=cache_dir,
            xs_table_path=xs_table_path,
            cache_filename="psg_o3_chappuis.txt",
        )


__all__: list[Any] = [
    "O3_MOLMASS",
    "PSG_O3_URL",
    "O3ChappuisOpacity",
    "_load_psg_xs_table",
    "fetch_psg_o3",
]
