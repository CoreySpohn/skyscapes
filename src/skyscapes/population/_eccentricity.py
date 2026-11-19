"""Eccentricity distributions of planet populations, as `priorities` records."""

import priorities

_KIPPING_A = 0.867
_KIPPING_B = 3.03


def kipping_eccentricity(e_max=None):
    """Kipping (2013) Beta(0.867, 3.03) eccentricity distribution.

    Args:
        e_max: Upper cut of the eccentricity, or ``None`` for no cut. A cut
            conditions the Beta law on ``e <= e_max`` through
            `priorities.TruncatedDistribution`.

    Returns:
        A `priorities.Beta`, or a `priorities.TruncatedDistribution` of it when
        `e_max` is given.
    """
    beta = priorities.Beta(_KIPPING_A, _KIPPING_B)
    return beta if e_max is None else priorities.TruncatedDistribution(beta, high=e_max)


def rayleigh_eccentricity(scale, e_max=None):
    """Rayleigh eccentricity distribution of mode `scale`.

    Args:
        scale: Mode of the density (0.049 is the Van Eylen et al. 2019 value
            for compact multi-planet systems).
        e_max: Upper cut of the eccentricity, or ``None`` for no cut.

    Returns:
        A `priorities.Rayleigh`, or a `priorities.TruncatedDistribution` of it
        when `e_max` is given.
    """
    rayleigh = priorities.Rayleigh(scale)
    return (
        rayleigh
        if e_max is None
        else priorities.TruncatedDistribution(rayleigh, high=e_max)
    )
