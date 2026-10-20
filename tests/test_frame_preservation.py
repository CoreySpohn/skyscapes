"""Physical disk images and demo elements survive the public frame change."""

import jax.numpy as jnp
import numpy as np
import pytest

from skyscapes.disk._los import los_integrate_scattered
from skyscapes.io import from_exovista

# Captured from skyscapes 51c19bc with orbix c8890bb, CPU float64.
# These are complete 5x5 images, with forward scattering and both tilt signs.
LOS_IMAGES = [
    [
        [
            0.005901539845819418,
            0.014003996723827346,
            0.014604248097375783,
            0.007743103851252473,
            0.0014786317096044934,
        ],
        [
            0.029673394058523345,
            0.04394652869221852,
            0.03542064589536184,
            0.031401754474588775,
            0.01311497076368116,
        ],
        [
            0.05807114874781169,
            0.09293530901214446,
            7.900181400463259e-08,
            0.04938505042705297,
            0.03131161527027989,
        ],
        [
            0.04570035969871473,
            0.14539339396592627,
            0.174411075915994,
            0.09462078559263547,
            0.03654507636588042,
        ],
        [
            0.006655069316704764,
            0.03962984459843626,
            0.06978137519968405,
            0.04597430701616469,
            0.0124871308386568,
        ],
    ],
    [
        [
            3.5228815608378613e-15,
            0.0008776497964583124,
            0.06026618222595367,
            0.07378508662688307,
            0.013881770191259229,
        ],
        [
            0.00020119309487552292,
            0.16287516676596886,
            0.4762272973737917,
            0.0873509173894697,
            0.013321886567472472,
        ],
        [
            0.031077664359012474,
            0.5451908388021246,
            0.05100301527140083,
            0.04095019269244589,
            0.0025097242776169937,
        ],
        [
            0.0641122232398493,
            0.12977268157736452,
            0.046181989766337216,
            0.00876361971817007,
            1.336207637174197e-05,
        ],
        [
            0.020165220964221492,
            0.027684306928022213,
            0.006308324784235689,
            5.978317948960136e-05,
            3.8181976728451653e-16,
        ],
    ],
    [
        [
            2.3867044261225546e-09,
            0.004620539996738293,
            0.055040757959668535,
            0.0627040663807771,
            0.017472860687240858,
        ],
        [
            0.007621473403467968,
            0.22129532410508526,
            0.41137695847151906,
            0.10800476969914954,
            0.02776670633834092,
        ],
        [
            0.07794456657366981,
            0.347576554445451,
            0.013867658057832983,
            0.04444419430046226,
            0.01059461383436695,
        ],
        [
            0.06350748024999214,
            0.07781032712953634,
            0.040644446480077645,
            0.015298612376041477,
            0.000617786602977209,
        ],
        [
            0.012776601307484207,
            0.016243110939192323,
            0.005776992272812416,
            0.0003470285012099818,
            2.6915983237661764e-10,
        ],
    ],
]

# All eight planets in the canonical demo FITS, in file order, from the same
# versions: a_AU, e, i_rad, W_rad, w_rad, M0_rad. Angles/elements must not change.
DEMO_ELEMENTS = [
    [
        0.387099,
        0.205625,
        1.130700508630296,
        2.195196872316272,
        4.447279311085948,
        3.0507260392391107,
    ],
    [
        0.722333199,
        0.00677323,
        1.061791299413483,
        2.1604296332674533,
        5.408989012265763,
        0.8761400741357597,
    ],
    [
        1.00000011,
        0.01671022,
        1.0471984070615956,
        2.0943949056238194,
        4.938259139713113,
        6.239952621485405,
    ],
    [
        1.52366231,
        0.09341233,
        1.0683109208837305,
        2.1224514048823475,
        2.7098146614955705,
        0.3386505464053476,
    ],
    [
        5.20336301,
        0.04839266,
        1.0431694355595715,
        2.120317184670284,
        3.386376544792743,
        0.3426787099262492,
    ],
    [
        9.53707032,
        0.0541506,
        1.03022203168904,
        2.140699654327287,
        4.730829565449819,
        5.542146144065518,
    ],
    [
        19.19126393,
        0.04716771,
        1.0508973908795851,
        2.1092947244136346,
        6.118510228872223,
        2.4825887001232516,
    ],
    [
        30.06896348,
        0.00858587,
        1.0268049688944991,
        2.1213289288422223,
        3.908464138773517,
        4.54058365372443,
    ],
]


@pytest.mark.parametrize(
    ("index", "inc", "pa"), [(0, 35.0, 20.0), (1, 60.0, 130.0), (2, 125.0, -40.0)]
)
def test_disk_image_mirrors_stored_columns(index, inc, pa):
    """The FITS column convention mirrors stored images, preserving the near half."""
    image = los_integrate_scattered(
        lambda r, z, valid: jnp.exp(-((r - 2.0) ** 2 + (z / 0.2) ** 2)),
        lambda cos_phi: (1.0 - 0.6**2) / (1.0 + 0.6**2 - 1.2 * cos_phi) ** 1.5,
        incl_deg=jnp.array(inc),
        pa_deg=jnp.array(pa),
        rmin_AU=jnp.array(0.5),
        rmax_AU=jnp.array(4.0),
        zmax_AU=jnp.array(1.0),
        nx=5,
        ny=5,
        pixel_scale_arcsec=0.1,
        dist_pc=10.0,
        n_slices_los=101,
    )
    # Mirror the stored arrays for +column west; do not recapture snapshots.
    expected = np.asarray(LOS_IMAGES[index])[:, ::-1]
    np.testing.assert_allclose(image, expected, rtol=1e-12, atol=0)
    y, col = np.indices(image.shape) - 2
    x = -col  # Sky east is minus the FITS column offset.
    minor = -x * np.sin(np.radians(pa)) + y * np.cos(np.radians(pa))
    assert np.sum(image * minor) * np.cos(np.radians(inc)) > 0


def test_demo_elements_unchanged(fits_fixture):
    """Changing vector components preserves the demo's recovered orbital elements."""
    system = from_exovista(fits_fixture, load_disk=False)
    actual = [
        [
            float(getattr(p.orbit, key)[0])
            for key in ("a_AU", "e", "i_rad", "W_rad", "w_rad", "M0_rad")
        ]
        for p in system.planets
    ]
    np.testing.assert_allclose(actual, DEMO_ELEMENTS, rtol=1e-12, atol=1e-14)
