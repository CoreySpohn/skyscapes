"""Occurrence-rate grid and planet-type table behind the planet population draw.

The grid is built from the Dulz et al. (2020) nominal occurrence-rate tables
(mass and radius versions) the way the ExoVista planet generator builds it:
the radius table covers planets smaller than 10 Earth radii, the mass table
the giants, and each table is convolved with the Chen and Kipping (2017)
mass-radius scatter to fill the other's regime. The result is a grid of
expected planets per star in cells of (scaled semi-major axis, mass), where
the scaled semi-major axis is ``a / sqrt(L)`` in AU for a host of ``L`` solar
luminosities. Cell edges are refined at the radius and stellar-flux bin edges of
``planetbins.dat`` so that every cell lies inside one planet-type bin.
"""

import csv
import gzip
import io
from functools import cache
from importlib import resources

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm

_R_SPLIT = 10.0  # Earth radii: the radius table below, the mass table above
_M_SPLIT = (_R_SPLIT / (1.008 * 2.04**0.279)) ** (1 / 0.589) * 2.04
_M_BREAK_1, _M_BREAK_2 = 2.04, 131.6  # Chen and Kipping (2017) regime breaks
_R_AT_BREAK_1 = 1.008 * _M_BREAK_1**0.279
_R_AT_BREAK_2 = _R_AT_BREAK_1 * (_M_BREAK_2 / _M_BREAK_1) ** 0.589


def mass_to_radius(mass_mearth):
    """Mean radius in Earth radii of a planet, Chen and Kipping (2017) forecaster.

    Accepts NumPy or JAX input and returns the same kind.
    """
    xp = jnp if isinstance(mass_mearth, jax.Array) else np
    m = xp.asarray(mass_mearth, float)
    return xp.where(
        m <= _M_BREAK_1,
        1.008 * m**0.279,
        xp.where(
            m <= _M_BREAK_2,
            _R_AT_BREAK_1 * (m / _M_BREAK_1) ** 0.589,
            _R_AT_BREAK_2 * (m / _M_BREAK_2) ** -0.044,
        ),
    )


def radius_to_mass(radius_rearth):
    """Mass in Earth masses for a radius, the inverse of `mass_to_radius`.

    The Jovian branch is not monotonic; like the generator this returns the
    lower-mass branch, and it is only used for radius bin edges below 10 Earth radii.
    """
    r = np.asarray(radius_rearth, float)
    mass = (r / 1.008) ** (1.0 / 0.279)
    mass = np.where(
        r > _R_AT_BREAK_1, (r / _R_AT_BREAK_1) ** (1.0 / 0.589) * _M_BREAK_1, mass
    )
    top = _R_AT_BREAK_1 * 131.6**0.589
    return np.where(r > top, (r / _R_AT_BREAK_2) ** (1.0 / -0.044) * _M_BREAK_2, mass)


def _read_gz_csv(name):
    """Columns of a gzipped numeric CSV as arrays keyed by header (spaces removed)."""
    raw = resources.files("skyscapes.population").joinpath("data", name).read_bytes()
    text = gzip.decompress(raw).decode()
    header = [h.replace(" ", "") for h in text.splitlines()[0].split(",")]
    values = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1)
    return {h: values[:, k] for k, h in enumerate(header)}


def _read_text(name):
    return resources.files("skyscapes.population").joinpath("data", name).read_text()


def _read_planet_bins():
    """Radius bin edges and, per radius bin, scaled semi-major-axis edges in AU."""
    lines = _read_text("planetbins.dat").splitlines()
    if lines[0].strip() != "Radius" or lines[2].strip() != "Flux":
        raise ValueError("unexpected planetbins.dat layout")
    radius_edges = np.array(lines[1].split(), float)
    flux_edges = np.array([row.split() for row in lines[3:] if row.strip()], float)
    return radius_edges, 1.0 / np.sqrt(flux_edges)


def _mid_and_edges(centers):
    """Log-spaced bin midpoints (as given) and their edges, extended at both ends."""
    edge = np.empty(centers.size + 1)
    edge[0] = centers[0] * np.sqrt(centers[0] / centers[1])
    edge[-1] = centers[-1] * np.sqrt(centers[-1] / centers[-2])
    edge[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    return edge


def _log_step(values, label):
    step = np.diff(np.log(values))
    if np.abs(step.max() - step.min()) / np.abs(step.min()) > 0.01:
        raise ValueError(
            f"{label} spacing of the occurrence tables is not constant in ln"
        )
    return step[0]


def _build_grid():
    """Occurrence grid refined at the planet-bin edges (the usebins branch)."""
    r_bound, a_bound = _read_planet_bins()
    table_m, table_r = (
        _read_gz_csv("NominalOcc_Mass.csv.gz"),
        _read_gz_csv("NominalOcc_Radius.csv.gz"),
    )
    occ_m, occ_r = table_m["Occ"], table_r["Occ"]
    a_mid = table_m["a_mid"]
    m_min, m_mid, m_max = (table_m[k] for k in ("M_min", "M_mid", "M_max"))
    r_min, r_mid, r_max = (table_r[k] for k in ("R_min", "R_mid", "R_max"))

    m_vec, r_vec, a_vec = np.unique(m_mid), np.unique(r_mid), np.unique(a_mid)
    dlnm, dlnr, dlna = (
        _log_step(m_vec, "mass"),
        _log_step(r_vec, "radius"),
        _log_step(a_vec, "semi-major axis"),
    )
    npix = a_vec.size
    m_bins = np.exp(np.linspace(np.log(m_vec[0]), np.log(m_vec[-1]), npix))
    r_bins = np.exp(np.linspace(np.log(r_vec[0]), np.log(r_vec[-1]), npix))
    a_bins = np.exp(np.linspace(np.log(a_vec[0]), np.log(a_vec[-1]), npix))

    m_edge_t = np.sort(
        np.concatenate(
            [_mid_and_edges(m_bins), [_M_SPLIT], radius_to_mass(r_bound[1:-1])]
        )
    )
    m_mid_t = np.sqrt(m_edge_t[:-1] * m_edge_t[1:])
    r_edge_t = np.sort(
        np.concatenate([_mid_and_edges(r_bins), [_R_SPLIT], r_bound[1:-1]])
    )
    r_mid_t = np.sqrt(r_edge_t[:-1] * r_edge_t[1:])
    r_mid_alt, r_edge_alt = mass_to_radius(m_mid_t), mass_to_radius(m_edge_t)
    m_mid_alt, m_edge_alt = radius_to_mass(r_mid_t), radius_to_mass(r_edge_t)
    a_edge = np.sort(
        np.concatenate([_mid_and_edges(a_bins), np.unique(a_bound[:, 1:-1].ravel())])
    )
    a_mid_out = np.sqrt(a_edge[:-1] * a_edge[1:])
    a_size = a_edge[1:] / a_edge[:-1]

    r_split_i = np.where(r_mid_t >= _R_SPLIT)[0][0]
    m_split_i = np.where(m_mid_t >= _M_SPLIT)[0][0]
    m_edge = np.concatenate((m_edge_alt[:r_split_i], m_edge_t[m_split_i:]))
    m_mid_out = np.concatenate((m_mid_alt[:r_split_i], m_mid_t[m_split_i:]))
    r_edge = np.concatenate((r_edge_t[:r_split_i], r_edge_alt[m_split_i:]))
    r_mid_out = np.concatenate((r_mid_t[:r_split_i], r_mid_alt[m_split_i:]))
    r_size, m_size = r_edge[1:] / r_edge[:-1], m_edge[1:] / m_edge[:-1]

    pad = 10
    occ_r_pad = np.zeros((a_vec.size, r_vec.size + 2 * pad))
    occ_m_pad = np.zeros((a_vec.size, m_vec.size + 2 * pad))
    ia = np.searchsorted(a_vec, a_mid)
    im = np.searchsorted(m_vec, m_mid)
    ir = np.searchsorted(r_vec, r_mid)
    occ_r_pad[ia, ir + pad] = occ_r
    occ_m_pad[ia, im + pad] = occ_m
    occ_r_pad[:, :pad] = occ_r_pad[:, [pad]]
    occ_r_pad[:, -pad:] = occ_r_pad[:, [-pad - 1]]
    occ_m_pad[:, :pad] = occ_m_pad[:, [pad]]
    occ_m_pad[:, -pad:] = occ_m_pad[:, [-pad - 1]]

    def padded(vec, step):
        out = np.zeros(vec.size + 2 * pad)
        out[pad:-pad] = vec
        for i in range(pad):
            out[pad - 1 - i] = out[pad - i] / np.exp(step)
            out[-pad + i] = out[-pad - 1 + i] * np.exp(step)
        return out

    r_vec_p, r_min_p, r_max_p = (
        padded(np.unique(r), dlnr) for r in (r_mid, r_min, r_max)
    )
    m_vec_p, m_min_p, m_max_p = (
        padded(np.unique(m), dlnm) for m in (m_mid, m_min, m_max)
    )

    cv_r = np.zeros((a_vec.size, r_vec.size))
    cv_m = np.zeros((a_vec.size, m_vec.size))
    for i in range(r_vec.size):
        if r_vec[i] < _R_SPLIT:
            cv_r[:, i] = occ_r_pad[:, i + pad]
        if m_vec[i] > _M_SPLIT:
            cv_m[:, i] = occ_m_pad[:, i + pad]

    new_r_pad = np.log(mass_to_radius(m_vec_p))
    new_m_pad = np.log(radius_to_mass(r_vec_p))
    sig_r = np.where(
        m_vec_p <= np.log(2.04),
        np.log(1.0403),
        np.where(m_vec_p <= np.log(131.6), np.log(1.1460), np.log(1.0737)),
    )
    # Fill the radius grid at 10 Earth radii and above from the mass table.
    r_lo, r_hi = np.log(r_min_p), np.log(r_max_p)
    for i in range(m_vec_p.size):
        if np.exp(new_r_pad[i]) < _R_SPLIT:
            continue
        w = np.abs(
            norm.cdf(r_hi, new_r_pad[i], sig_r[i])
            - norm.cdf(r_lo, new_r_pad[i], sig_r[i])
        )
        cv_r += occ_m_pad[:, [i]] * w[None, pad:-pad]
    sig_m = np.where(
        new_m_pad <= np.log(2.04),
        np.log(1.0403) / 0.279,
        np.where(
            new_m_pad <= np.log(131.6), np.log(1.1460) / 0.589, np.log(1.0737) / 0.044
        ),
    )
    m_lo, m_hi = np.log(m_min_p), np.log(m_max_p)
    for i in range(r_vec_p.size):
        if np.exp(new_m_pad[i]) > _M_SPLIT:
            continue
        w = np.abs(
            norm.cdf(m_hi, new_m_pad[i], sig_m[i])
            - norm.cdf(m_lo, new_m_pad[i], sig_m[i])
        )
        cv_m += occ_r_pad[:, [i]] * w[None, pad:-pad]

    cv_r /= dlnr * dlna
    cv_m /= dlnm * dlna
    ln_a = np.log(a_vec)
    interp_r = RectBivariateSpline(np.log(r_vec), ln_a, cv_r.T)
    interp_m = RectBivariateSpline(np.log(m_vec), ln_a, cv_m.T)
    final_split = np.where(r_mid_out >= _R_SPLIT)[0][0]
    occ = np.zeros((a_mid_out.size, m_mid_out.size))
    occ[:, :final_split] = interp_r(
        np.log(r_mid_out[:final_split]), np.log(a_mid_out)
    ).T
    occ[:, final_split:] = interp_m(
        np.log(m_mid_out[final_split:]), np.log(a_mid_out)
    ).T
    occ = occ * (occ > 0)
    occ[:, :final_split] *= np.outer(np.log(a_size), np.log(r_size))[:, :final_split]
    occ[:, final_split:] *= np.outer(np.log(a_size), np.log(m_size))[:, final_split:]
    bin_id = _bin_ids(occ.shape, r_bound, a_bound, m_edge, m_mid_out, a_mid_out)
    return occ, m_edge, a_edge, bin_id


def _bin_ids(shape, r_bound, a_bound, m_edge, m_mid, a_mid):
    """Planet-bin index of every cell, -1 for cells outside all bins.

    A bin is one radius bin of ``planetbins.dat`` crossed with one stellar-flux
    sub-bin; a cell belongs to the bin whose edges contain its midpoint.
    """
    m_bound = radius_to_mass(r_bound)
    m_bound[-1] = m_edge[-1]
    bin_id = np.full(shape, -1, np.int64)

    def first_at_or_above(values, edge, default):
        hits = np.where(values >= edge)[0]
        return int(hits[0]) if hits.size else default

    count = 0
    for i in range(r_bound.size - 1):
        lo = first_at_or_above(m_mid, m_bound[i], 0)
        hi = first_at_or_above(m_mid, m_bound[i + 1], m_mid.size)
        for j in range(a_bound.shape[1] - 1):
            a_lo = first_at_or_above(a_mid, a_bound[i, j], 0)
            a_hi = first_at_or_above(a_mid, a_bound[i, j + 1], a_mid.size)
            bin_id[a_lo:a_hi, lo:hi] = count
            count += 1
    return bin_id


@cache
def _grid():
    return _build_grid()


class OccurrenceTable(eqx.Module):
    """Expected planets per star in (scaled semi-major axis, mass) cells.

    Attributes:
        rate: ``float[n_a, n_m]`` expected planets per star in each cell.
        a_edges_au: ``float[n_a + 1]`` cell edges of ``a / sqrt(L)`` in AU.
        mass_edges_mearth: ``float[n_m + 1]`` cell edges of planet mass in Earth masses.
        bin_id: ``int[n_a, n_m]`` planet bin of each cell (a radius bin crossed with a
            stellar-flux sub-bin); the draw fixes the planet count of each bin.
        n_bins: Number of planet bins.
        total_rate: Expected planets per star over the binned cells.
    """

    rate: jnp.ndarray
    a_edges_au: jnp.ndarray
    mass_edges_mearth: jnp.ndarray
    bin_id: jnp.ndarray
    n_bins: int = eqx.field(static=True)
    total_rate: float = eqx.field(static=True)


def occurrence_table_from_grid(rate, a_edges_au, mass_edges_mearth, bin_id):
    """Build an `OccurrenceTable` from a rate grid and the bin index of each cell.

    Args:
        rate: ``[n_a, n_m]`` expected planets per star per cell.
        a_edges_au: ``[n_a + 1]`` edges of the scaled semi-major axis.
        mass_edges_mearth: ``[n_m + 1]`` edges of the planet mass.
        bin_id: ``int[n_a, n_m]`` planet bin of each cell, -1 outside every bin.
    """
    rate, bin_id = np.asarray(rate, float), np.asarray(bin_id, np.int32)
    return OccurrenceTable(
        rate=jnp.asarray(rate),
        a_edges_au=jnp.asarray(a_edges_au),
        mass_edges_mearth=jnp.asarray(mass_edges_mearth),
        bin_id=jnp.asarray(bin_id),
        n_bins=int(bin_id.max()) + 1,
        total_rate=float(rate[bin_id >= 0].sum()),
    )


def load_occurrence_table():
    """The Dulz et al. (2020) nominal occurrence grid, as built by ExoVista."""
    rate, m_edge, a_edge, bin_id = _grid()
    return occurrence_table_from_grid(rate, a_edge, m_edge, bin_id)


class PlanetTypes(eqx.Module):
    """Planet types, the ranges that select them and their albedo spectra.

    Attributes:
        names: One name per type.
        radius_min_rearth: ``[n_types]`` lower radius edge; for exoEarth-candidate
            types the edge is scaled by ``1 / sqrt(a_scaled)``.
        radius_max_rearth: ``[n_types]`` upper radius edge.
        a_min_au: ``[n_types]`` lower edge of the scaled semi-major axis.
        a_max_au: ``[n_types]`` upper edge of the scaled semi-major axis.
        weight: ``[n_types]`` selection weight among the types that fit a planet.
        earth_candidate: ``bool[n_types]`` whether the type is an exoEarth candidate.
        wavelength_nm: ``[n_wavelengths]`` common wavelength grid of the spectra.
        albedo: ``[n_types, n_wavelengths]`` geometric albedo spectra.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    radius_min_rearth: jnp.ndarray
    radius_max_rearth: jnp.ndarray
    a_min_au: jnp.ndarray
    a_max_au: jnp.ndarray
    weight: jnp.ndarray
    earth_candidate: jnp.ndarray
    wavelength_nm: jnp.ndarray
    albedo: jnp.ndarray

    def geometric_albedo(self, type_index, wavelength_nm):
        """Geometric albedo of the planets of `type_index` at one wavelength."""
        spectra = jax.vmap(lambda y: jnp.interp(wavelength_nm, self.wavelength_nm, y))(
            self.albedo
        )
        return spectra[type_index]


def load_planet_types():
    """The planet types of the ExoVista albedo list with their albedo spectra."""
    rows = list(csv.reader(io.StringIO(_read_text("albedo_list.csv"))))
    rows = [[field.strip() for field in row] for row in rows[1:] if row]
    names = tuple(row[0] for row in rows)

    def column(k):
        return jnp.asarray([float(row[k]) for row in rows])

    spectra = np.load(
        resources.files("skyscapes.population").joinpath("data", "albedo_spectra.npz")
    )
    if tuple(str(n) for n in spectra["names"]) != names:
        raise ValueError("albedo_spectra.npz does not match albedo_list.csv")
    return PlanetTypes(
        names=names,
        radius_min_rearth=column(1),
        radius_max_rearth=column(2),
        a_min_au=column(3),
        a_max_au=column(4),
        weight=column(5),
        earth_candidate=jnp.asarray([row[6] == "True" for row in rows]),
        wavelength_nm=jnp.asarray(spectra["wavelength_nm"]),
        albedo=jnp.asarray(spectra["albedo"]),
    )
