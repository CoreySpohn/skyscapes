"""β-axis integrity: scene contrast queries return the stored β-indexed grid.

After the β-axis fix in ``skyscapes.io.exovista._load_single_planet``,
``GridAtmosphere.phase_angle_deg`` must actually be phase angle β, and
queries at propagated β should return physically sensible contrast values
on [0, 180], not extrapolated garbage outside the grid.
"""

from __future__ import annotations

import jax.numpy as jnp


def test_phase_axis_is_beta_not_mean_anomaly(scene_system):
    """Each planet's atmosphere axis spans β ∈ [0, 180], not M ∈ [0, 360]."""
    for planet in scene_system.planets:
        phase_axis = planet.atmosphere.phase_angle_deg
        assert float(phase_axis.min()) >= 0.0
        assert float(phase_axis.max()) <= 180.0 + 1e-3, (
            f"phase_angle_deg max {float(phase_axis.max())} exceeds 180° — "
            "axis is still mean anomaly, not β"
        )


def test_scene_contrast_finite_and_positive(scene_system):
    """Querying scene.Planet.contrast at propagated times returns finite β-lookups."""
    star = scene_system.star
    solver = scene_system.trig_solver
    t0 = float(star._times_jd[0])
    t_query = jnp.array([t0, t0 + 50.0, t0 + 150.0])
    wl_query = jnp.array([500.0, 700.0, 900.0])

    for planet in scene_system.planets[: min(3, len(scene_system.planets))]:
        c = planet.contrast(solver, wl_query, t_query, star=star)
        assert jnp.isfinite(c).all(), "contrast has NaN/Inf"
        assert (c > 0).all(), "contrast is non-positive"
        assert (c < 1.0).all(), "contrast exceeds unity (unphysical)"
