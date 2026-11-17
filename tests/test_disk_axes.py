"""The disk viz basis against an independent astronomical construction."""

import numpy as np
import pytest


@pytest.mark.parametrize(("inc", "pa"), [(35.0, 20.0), (60.0, 130.0), (125.0, -40.0)])
def test_disk_axes_preserve_physical_plane_and_near_half(inc, pa):
    """Independent astronomical basis, with the near half at negative z.

    The position angle runs north through east, so the line of nodes in
    (east, north, away) is (sin pa, cos pa, 0). The near in-plane axis is the
    nodes turned 90 degrees toward west of north, (-cos pa, sin pa, 0), tilted
    by the inclination toward the observer; the normal is nodes x near.
    """
    from skyscapes.viz._geometry import disk_axes_sky

    i, p = np.radians([inc, pa])
    nodes = np.array([np.sin(p), np.cos(p), 0.0])
    near = np.array([-np.cos(p) * np.cos(i), np.sin(p) * np.cos(i), -np.sin(i)])
    normal = np.cross(nodes, near)
    np.testing.assert_allclose(
        disk_axes_sky(inc, pa), [nodes, near, normal], atol=1e-14
    )
    assert near[2] < 0
