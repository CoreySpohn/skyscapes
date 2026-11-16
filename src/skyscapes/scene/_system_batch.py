"""Host packing and fixed padding for scene System pytrees."""

import equinox as eqx
import jax
import numpy as np
from hwoutils.epoch import Epoch
from orbix.system.core import GM_SUN_AU3_PER_D2

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
        "nu_grid",
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


def _planet(radius, model):
    """Construct without transferring host radius data through JAX."""
    return eqx.tree_at(lambda p: p.Rp_Rearth, Planet(np.empty(0), model), radius)


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
        return _planet(np.empty(0), model)
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
        jax.tree.unflatten(definition, merged),
    )


def from_systems(cls, systems, *, capacity=None, batch_shape=None):
    """Pack unbatched systems, sorted by semi-major axis, on the host."""
    systems = list(systems)
    if not systems:
        raise ValueError("systems must be nonempty")
    if any(s.batch_shape for s in systems):
        raise ValueError("from_systems requires unbatched systems")
    if any(s.disk is not None for s in systems):
        raise ValueError("Systems with a disk cannot enter a batch")
    if any(s.epoch != systems[0].epoch for s in systems):
        raise ValueError("Systems must share one epoch")
    star_classes = {type(s.star) for s in systems}
    model_classes = {type(p.physical_model) for s in systems for p in s.planets}
    if len(star_classes) != 1 or len(model_classes) != 1:
        raise ValueError("Systems must share one star class and physical-model class")
    model_template = max(
        (p for s in systems for p in s.planets), key=lambda p: p.n_planets
    )
    template_leaves = dict(
        (jax.tree_util.keystr(path), x)
        for path, x in jax.tree_util.tree_flatten_with_path(
            model_template.physical_model
        )[0]
    )
    counts = [int(np.count_nonzero(s.valid)) for s in systems]
    capacity = max(counts) if capacity is None else capacity
    if not isinstance(capacity, int) or capacity < max(counts):
        raise ValueError(
            "capacity must be an integer at least the largest planet count"
        )
    batch_shape = (len(systems),) if batch_shape is None else tuple(batch_shape)
    if np.prod(batch_shape, dtype=int) != len(systems):
        raise ValueError("batch_shape must contain exactly len(systems) entries")
    rows = []
    for system, count in zip(systems, counts, strict=True):
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
            (_planet(radius, model),),
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
        raise ValueError("Systems must share model classes and static structure")
    return jax.tree.map(
        lambda *xs: np.stack(xs).reshape((*batch_shape, *np.shape(xs[0]))), *rows
    )


def to_systems(system):
    """Unpack host scenes, discarding padding and restoring source leaf order."""
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
            planets.append(_planet(group.Rp_Rearth[slots], model))
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
    """Flatten every dynamic leaf under its JAX key path, with epoch and K."""
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


def from_table(cls, table, *, template):
    """Restore leaf arrays using a template for model classes and static fields."""
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
    frame = type(skeleton.dynamics.frame)(skeleton.dist_pc, epoch)
    skeleton = eqx.tree_at(lambda s: s.dynamics.frame, skeleton, frame)
    paths, definition = jax.tree_util.tree_flatten_with_path(skeleton)
    restored = jax.tree.unflatten(
        definition,
        [
            np.asarray(table[jax.tree_util.keystr(path).lstrip(".")]).copy()
            for path, _ in paths
        ],
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
