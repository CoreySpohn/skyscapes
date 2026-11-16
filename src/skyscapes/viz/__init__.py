"""Scene visualizations in east/north/away sky coordinates.

Disk major-axis position angles are astronomical, north through east.
Inclination is zero face-on. Photon radiance images carry their PixelGrid.
"""

import importlib

_LAZY = {
    "plot_disk_geometry": "skyscapes.viz.disk",
    "plot_disk_radiance": "skyscapes.viz.disk",
    "plot_local_zodi_geometry": "skyscapes.viz.zodi",
    "plot_scattering_angle": "skyscapes.viz.scattering",
    "plot_system": "skyscapes.viz.system",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve a lazy re-export, checking the eyepiece requirement first.

    Args:
        name: Attribute being looked up on ``skyscapes.viz``.

    Returns:
        The requested plot function.

    Raises:
        AttributeError: If ``name`` is not one of the lazy re-exports.
    """
    if name in _LAZY:
        from skyscapes.viz import _require

        _require.eyepiece()
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'skyscapes.viz' has no attribute {name!r}")


def __dir__():
    """List the lazy re-exports alongside the module's real attributes.

    Returns:
        Sorted attribute names, including the lazily provided functions.
    """
    return sorted(set(globals()) | set(__all__))
