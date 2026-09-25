"""Call-time color resolution for the views.

Nothing here is bound at import time: a mode switch between two calls is
honored because every color is looked up when a view is drawn.
"""

from __future__ import annotations

import numpy as np


def _mode():
    import hwostyle

    return hwostyle.current_mode()


def role(name):
    """The active mode's brand color for ``star``, ``planet`` or ``disk``."""
    import hwostyle

    if _mode():
        return hwostyle.roles[name]
    from hwostyle.palettes import Roles

    return Roles("light")[name]


def local_zodi():
    """The green palette slot, the color of local zodiacal light.

    hwostyle carries no background role yet, so this names the palette
    color the figure conventions assign to sky backgrounds.
    """
    import hwostyle
    from hwostyle.palettes import Palette

    palette = hwostyle.palette if _mode() else Palette("light")
    colors = palette.as_dict
    return colors.get("green", palette[3 % len(palette)])


def text():
    """The mode's text color, the rank reserved for the answer."""
    import matplotlib as mpl

    return mpl.rcParams["text.color"]


def neutral(level):
    """A tone ``level`` of the way from the axes facecolor to the text color.

    Scenery (reference lines, the sky plane, leader labels) takes a neutral
    tone that inverts with the mode rather than one frozen gray.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_rgb

    face = np.asarray(to_rgb(mpl.rcParams["axes.facecolor"]))
    fore = np.asarray(to_rgb(mpl.rcParams["text.color"]))
    return tuple(face + level * (fore - face))
