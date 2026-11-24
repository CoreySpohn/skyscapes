"""Host packing and fixed padding for Scene pytrees."""

import dataclasses

import equinox as eqx
import jax
import numpy as np
from hwoutils.constants import GM_SUN_AU3_PER_D2
from hwoutils.epoch import Epoch
from optixstuff.spectral import same_wavelength

from .planet import Planet

# One circular, face-on 1 AU orbit, with zero phase and a fixed solar rate.
_PAD_ORBIT = dict(
    a_AU=np.asarray(1.0),
    n_rad_per_d=np.asarray(np.sqrt(GM_SUN_AU3_PER_D2)),
    ex=np.asarray(0.0),
    ey=np.asarray(0.0),
    quat=np.asarray([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0]),
    epoch_d=np.asarray(0.0),
)
_SHARED_GRIDS = frozenset(
    (
        "wavelengths_nm",
        "phase_angle_deg",
        "times_d",
        "spectrum",
        "rayleigh_xs",
        "xsmatrix_per_species",
        "ssa_grid",
        "g_grid",
        "cia",
    )
)


def _per_planet(path, value, count):
    """Distinguish spectral coordinates from leading planet parameter axes."""
    return (
        np.ndim(value) > 0
        and np.shape(value)[0] == count
        and not any(getattr(key, "name", None) in _SHARED_GRIDS for key in path)
    )


def _host(tree):
    return jax.tree.map(
        lambda x: np.asarray(
            x, dtype=bool if np.asarray(x).dtype == bool else np.float64
        ),
        tree,
    )


def _planet(radius, scale, model):
    """Construct without transferring host radius data through JAX."""
    return eqx.tree_at(
        lambda p: (p.Rp_Rearth, p.flux_scale),
        Planet(np.empty(0), model),
        (radius, scale),
    )


def _planet_group(system, model_template):
    """Combine same-model groups in scene order, retaining shared grids."""
    if not system.planets:
        model = jax.tree_util.tree_map_with_path(
            lambda path, x: (
                np.empty((0, *np.shape(x)[1:]))
                if _per_planet(path, x, model_template.n_planets)
                else np.asarray(x)
            ),
            model_template.physical_model,
        )
        return _planet(np.empty(0), np.empty(0), model)
    groups = system.planets
    definitions = [jax.tree.structure(p.physical_model) for p in groups]
    if any(t != definitions[0] for t in definitions):
        raise ValueError("Physical models must share one class and static structure")
    paths, definition = jax.tree_util.tree_flatten_with_path(groups[0].physical_model)
    leaves = [jax.tree.leaves(p.physical_model) for p in groups]
    merged = []
    for j, (path, x) in enumerate(paths):
        if _per_planet(path, x, groups[0].n_planets):
            merged.append(np.concatenate([np.asarray(row[j]) for row in leaves]))
        else:
            if any(not np.array_equal(x, row[j]) for row in leaves):
                raise ValueError("Planet groups must share spectral coordinates")
            merged.append(np.asarray(x))
    return _planet(
        np.concatenate([np.asarray(p.Rp_Rearth) for p in groups]),
        np.concatenate([np.asarray(p.flux_scale) for p in groups]),
        jax.tree.unflatten(definition, merged),
    )


def _check_star_knots(scenes):
    """Refuse star grids that differ, naming twins that differ only by rounding.

    Args:
        scenes: The scenes to pack.

    Raises:
        ValueError: If two stars hold different knots. When the grids agree
            within the wavelength tolerance (a float32 copy of a float64
            file grid, say), the message names both scenes and the fix.
    """
    first = getattr(scenes[0].star, "knots", None)
    for index, scene in enumerate(scenes[1:], start=1):
        knots = getattr(scene.star, "knots", None)
        if knots == first:
            continue
        if (
            first is not None
            and knots is not None
            and len(knots) == len(first)
            and bool(np.all(same_wavelength(knots.nm, first.nm)))
        ):
            raise ValueError(
                f"the stars of scene 0 and scene {index} have knots that agree "
                "within the wavelength tolerance but differ in digest "
                f"({first.digest} and {knots.digest}), so they cannot share one "
                "batch; build both stars on one shared Knots"
            )
        raise ValueError(
            f"the stars of scene 0 and scene {index} have different knots; a batch "
            "holds one star grid"
        )


def from_scenes(cls, scenes, *, capacity=None, batch_shape=None):
    """Pack unbatched scenes, sorted by semi-major axis, on the host.

    Args:
        cls: The scene class to construct for each packed row.
        scenes: Unbatched scenes sharing one epoch, star class and
            physical-model class, with no disk or local zodi.
        capacity: Planet slots per scene. None uses the largest valid planet
            count among the scenes; extra slots are padding.
        batch_shape: Leading batch axes of the result. None uses
            ``(len(scenes),)``; its product must equal ``len(scenes)``.

    Returns:
        A batched scene on the host whose leaves carry ``batch_shape`` as
        leading axes.
    """
    scenes = list(scenes)
    if not scenes:
        raise ValueError("scenes must be nonempty")
    if any(s.batch_shape for s in scenes):
        raise ValueError("from_scenes requires unbatched scenes")
    if any(s.disk is not None or s.local_zodi is not None for s in scenes):
        raise ValueError("Scenes with a disk or local zodi cannot enter a batch")
    if len({s.ra_deg is None for s in scenes}) > 1:
        raise ValueError(
            "Scenes must all be placed on the sky (ra_deg and dec_deg) or all unplaced"
        )
    if any(s.epoch != scenes[0].epoch for s in scenes):
        raise ValueError("Scenes must share one epoch")
    star_classes = {type(s.star) for s in scenes}
    model_classes = {type(p.physical_model) for s in scenes for p in s.planets}
    if len(star_classes) != 1 or len(model_classes) != 1:
        raise ValueError("Scenes must share one star class and physical-model class")
    _check_star_knots(scenes)
    model_template = max(
        (p for s in scenes for p in s.planets), key=lambda p: p.n_planets
    )
    template_leaves = dict(
        (jax.tree_util.keystr(path), x)
        for path, x in jax.tree_util.tree_flatten_with_path(
            model_template.physical_model
        )[0]
    )
    counts = [int(np.count_nonzero(s.valid)) for s in scenes]
    capacity = max(counts) if capacity is None else capacity
    if not isinstance(capacity, int) or capacity < max(counts):
        raise ValueError(
            "capacity must be an integer at least the largest planet count"
        )
    batch_shape = (len(scenes),) if batch_shape is None else tuple(batch_shape)
    if np.prod(batch_shape, dtype=int) != len(scenes):
        raise ValueError("batch_shape must contain exactly len(scenes) entries")
    rows = []
    for system, count in zip(scenes, counts, strict=True):
        group = _planet_group(system, model_template)
        chain = np.argsort(np.asarray(system.dynamics.orbits.a_AU), kind="stable")
        inverse = np.argsort(system.body_index)
        scene = inverse[chain]
        scene = scene[np.asarray(system.valid)[scene]]
        chain = np.asarray(system.body_index, dtype=int)[scene]

        def pad_planet(path, x, system=system, scene=scene, count=count):
            x = np.asarray(x)
            if not _per_planet(path, x, system.n_planets):
                return x
            real = x[scene]
            # Reuse a finite physical row, avoiding singular thermal parameters.
            fill = real[:1]
            if not count:
                source = np.asarray(template_leaves[jax.tree_util.keystr(path)])
                fill = (
                    source[:1]
                    if source.shape[0]
                    else np.full(
                        (1, *x.shape[1:]),
                        300.0 if path[-1].name in ("teff_K", "T_eq_K") else 1.0,
                    )
                )
            return np.concatenate([real, np.repeat(fill, capacity - count, axis=0)])

        model = jax.tree_util.tree_map_with_path(pad_planet, group.physical_model)
        radius = np.zeros(capacity)
        radius[:count] = np.asarray(group.Rp_Rearth)[scene]
        scale = np.ones(capacity)
        scale[:count] = np.asarray(group.flux_scale)[scene]
        masses = np.zeros(capacity)
        masses[:count] = np.asarray(system.dynamics.planets.mass_Msun)[chain]
        orbit = jax.tree_util.tree_map_with_path(
            lambda path, x, chain=chain, count=count: np.concatenate(
                [
                    np.asarray(x)[chain],
                    np.broadcast_to(
                        _PAD_ORBIT[path[-1].name], (capacity - count, *np.shape(x)[1:])
                    ),
                ]
            ),
            system.dynamics.orbits,
        )
        dynamics = eqx.tree_at(
            lambda d: (d.planets.mass_Msun, d.orbits), system.dynamics, (masses, orbit)
        )
        packed = cls(
            system.star,
            (_planet(radius, scale, model),),
            dynamics,
            midplane_inc_deg=system.midplane_inc_deg,
            midplane_position_angle_deg=system.midplane_position_angle_deg,
            valid=np.arange(capacity) < count,
        )
        # Fixed-size host layout data preserves source order and group boundaries.
        layout = np.full((3, capacity + 1), -1.0)
        layout[0, :count], layout[1, :count] = scene, chain
        group_ids = np.repeat(
            np.arange(len(system.planets)), [p.n_planets for p in system.planets]
        )
        layout[2, :count] = group_ids[scene]
        layout[0, -1] = len(system.planets)
        packed = eqx.tree_at(
            lambda s: s._source_layout, packed, layout, is_leaf=lambda x: x is None
        )
        rows.append(_host(packed))
    if any(jax.tree.structure(row) != jax.tree.structure(rows[0]) for row in rows):
        raise ValueError("Scenes must share model classes and static structure")
    return jax.tree.map(
        lambda *xs: np.stack(xs).reshape((*batch_shape, *np.shape(xs[0]))), *rows
    )


def to_scenes(system):
    """Unpack host scenes, discarding padding and restoring source leaf order.

    Args:
        system: A scene, batched or not.

    Returns:
        A list of unbatched host scenes, one per batch row in C order.
    """
    if (
        not system.batch_shape
        and system._source_layout is None
        and np.all(system.valid)
    ):
        return [_host(system)]
    result = []
    for index in np.ndindex(system.batch_shape):
        row = jax.tree.map(lambda x, index=index: np.asarray(x)[index], system)
        selected = np.flatnonzero(row.valid)
        layout = row._source_layout
        group = row.planets[0]
        if layout is None:
            scene_slots = np.asarray(row.body_index, dtype=int)[selected]
            chain_slots = np.sort(scene_slots)
            group_slots = [selected]
        else:
            chain_slots = selected[np.argsort(layout[1, selected], kind="stable")]
            scene_slots = selected[np.argsort(layout[0, selected], kind="stable")]
            group_slots = [
                scene_slots[layout[2, scene_slots] == i]
                for i in range(int(layout[0, -1]))
            ]
        planets = []
        for slots in group_slots:
            model = jax.tree_util.tree_map_with_path(
                lambda path, x, slots=slots, row=row: (
                    x[slots] if _per_planet(path, x, row.capacity) else x
                ),
                group.physical_model,
            )
            planets.append(
                _planet(group.Rp_Rearth[slots], group.flux_scale[slots], model)
            )
        dynamics = eqx.tree_at(
            lambda d: (d.planets.mass_Msun, d.orbits),
            row.dynamics,
            (
                row.dynamics.planets.mass_Msun[chain_slots],
                jax.tree.map(
                    lambda x, slots=chain_slots: x[slots], row.dynamics.orbits
                ),
            ),
        )
        # Packed scene slots and chain slots address the same sorted body array.
        mapping = {slot: i for i, slot in enumerate(chain_slots)}
        body_index = tuple(mapping[slot] for slot in scene_slots)
        result.append(
            _host(
                type(system)(
                    row.star,
                    tuple(planets),
                    dynamics,
                    midplane_inc_deg=row.midplane_inc_deg,
                    midplane_position_angle_deg=row.midplane_position_angle_deg,
                    body_index=body_index,
                )
            )
        )
    return result


def to_table(system):
    """Flatten every dynamic leaf under its JAX key path, with epoch and K.

    Args:
        system: A scene, batched or not.

    Returns:
        A dict of host arrays keyed by leaf path, plus ``body_index``, ``K``
        (the planet capacity) and the epoch entries ``epoch_jd1``,
        ``epoch_jd2`` and ``epoch_scale``.
    """
    table = {
        jax.tree_util.keystr(path).lstrip("."): np.asarray(x).copy()
        for path, x in jax.tree_util.tree_flatten_with_path(_host(system))[0]
    }
    table.update(
        body_index=np.asarray(system.body_index, dtype=int),
        K=np.asarray(system.capacity),
        epoch_jd1=np.asarray(system.epoch.jd1),
        epoch_jd2=np.asarray(system.epoch.jd2),
        epoch_scale=np.asarray(system.epoch.scale),
    )
    return table


def _table_leaf(table, path, template_leaf):
    """Read one leaf; a table written before ``flux_scale`` existed reads as ones."""
    key = jax.tree_util.keystr(path).lstrip(".")
    if key not in table and key.endswith("flux_scale"):
        radius = key.replace("flux_scale", "Rp_Rearth")
        shape = np.shape(table[radius]) if radius in table else np.shape(template_leaf)
        return np.ones(shape)
    return np.asarray(table[key]).copy()


def from_table(cls, table, *, template):
    """Restore leaf arrays using a template for model classes and static fields.

    Args:
        cls: The scene class to construct.
        table: A table written by :func:`to_table`.
        template: A scene supplying the model classes, static fields and
            structure; its leaf values are replaced by the table.

    Returns:
        The restored scene on the host.
    """
    capacity = int(table["K"])
    indices = {
        int(key[len("planets[") :].split("]", 1)[0])
        for key in table
        if key.startswith("planets[")
    }
    n_groups = max(indices, default=-1) + 1
    if indices != set(range(n_groups)):
        raise ValueError("Table planet groups must be contiguous")
    if n_groups and not template.planets:
        raise ValueError("A Planet template is required for model classes")
    groups = tuple(
        template.planets[min(i, len(template.planets) - 1)] for i in range(n_groups)
    )
    skeleton = eqx.tree_at(lambda s: s.planets, template, groups)
    skeleton = eqx.tree_at(
        lambda s: s._source_layout,
        skeleton,
        np.asarray(table["_source_layout"]) if "_source_layout" in table else None,
        is_leaf=lambda x: x is None,
    )
    epoch = Epoch(
        float(table["epoch_jd1"]), float(table["epoch_jd2"]), str(table["epoch_scale"])
    )
    # epoch is a static field, so it is replaced on the frame, keeping the position.
    frame = dataclasses.replace(skeleton.dynamics.frame, epoch=epoch)
    skeleton = eqx.tree_at(lambda s: s.dynamics.frame, skeleton, frame)
    paths, definition = jax.tree_util.tree_flatten_with_path(skeleton)
    restored = jax.tree.unflatten(
        definition,
        [_table_leaf(table, path, template_leaf) for path, template_leaf in paths],
    )
    # body_index is structural and must describe the stored capacity.
    if restored.capacity != capacity:
        raise ValueError("K must match the stored planet capacity")
    result = cls(
        restored.star,
        restored.planets,
        restored.dynamics,
        disk=restored.disk,
        midplane_inc_deg=restored.midplane_inc_deg,
        midplane_position_angle_deg=restored.midplane_position_angle_deg,
        valid=restored.valid,
        body_index=tuple(int(i) for i in table["body_index"]),
    )
    result = eqx.tree_at(
        lambda s: s._source_layout,
        result,
        restored._source_layout,
        is_leaf=lambda x: x is None,
    )
    return _host(result)
