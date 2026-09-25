"""Plotting for skyscapes types, built on eyepiece primitives.

Requires the ``viz`` extra (``pip install 'skyscapes[viz]'``), which brings
eyepiece and, through it, matplotlib and hwostyle. The base install stays
free of all three: names are re-exported lazily (PEP 562), so importing
this package imports no plotting stack, and the eyepiece requirement is
checked only when a plot function is first touched.

Every view draws in the library's own frames. The sky frame is
``(x, y, z)`` with ``x`` the RA offset (drawn increasing to the left), ``y``
the Dec offset and ``+z`` pointing from the star toward the observer, the
frame ``Planet.position_arcsec`` and the parametric disk kernels share. The
disk midplane is placed in that frame by ``System.midplane_inc_deg`` and
``System.midplane_pa_deg`` through the same rotation the ExoVista loader
uses, so a drawn outline and a rendered surface-brightness map agree.
That position angle turns the line of nodes from ``+x`` toward ``+y``; it
is not the astronomical position angle, which (with ``+x`` east) is
``90 - pa`` (mod 180) for the major axis. Inclinations run over
``[0, 180]`` in every view.
"""

import importlib

_LAZY = {
    "plot_disk_geometry": "skyscapes.viz.disk",
    "plot_disk_image": "skyscapes.viz.disk",
    "plot_local_zodi_geometry": "skyscapes.viz.zodi",
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
