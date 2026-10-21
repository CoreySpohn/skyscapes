---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
mystnb:
  execution_mode: force
---

# Get the habitable-zone edges of a star

{mod}`skyscapes.habitable_zone` returns the inner and outer edges of the
habitable zone, in AU, for one star or for a whole target list. It offers two
definitions.

- {func}`~skyscapes.habitable_zone.habitable_zone_au` scales fixed edges by
  the Earth-equivalent insolation distance (EEID), $a_\mathrm{EEID} =
  \sqrt{L_\star / L_\odot}$ AU. The conservative zone, 0.95 to 1.67 EEID, is
  the exoEarth-candidate zone of Stark et al. (2019); the optimistic zone,
  0.75 to 1.77 EEID, is the recent-Venus to early-Mars range of Kopparapu et
  al. (2013). It needs only the luminosity, and it is the definition used for
  exoEarth-candidate yields.
- {func}`~skyscapes.habitable_zone.kopparapu_habitable_zone_au` uses the
  effective-flux limits of Kopparapu et al. (2014), which also depend on the
  stellar effective temperature (valid for 2600 K to 7200 K).

## One star

```{code-cell} ipython3
import jax
import jax.numpy as jnp

from skyscapes.habitable_zone import (
    eeid_au,
    habitable_zone_au,
    kopparapu_habitable_zone_au,
)

jax.config.update("jax_enable_x64", True)

luminosity_lsun = 0.6
print("EEID [AU]:", float(eeid_au(luminosity_lsun)))
print("conservative [AU]:", [float(x) for x in habitable_zone_au(luminosity_lsun)])
print(
    "optimistic [AU]:",
    [float(x) for x in habitable_zone_au(luminosity_lsun, optimistic=True)],
)
print(
    "Kopparapu, Teff = 5200 K [AU]:",
    [float(x) for x in kopparapu_habitable_zone_au(luminosity_lsun, 5200.0)],
)
```

## A target list

Every function broadcasts over arrays and works under `jax.vmap`, so a target
list goes through in one call.

```{code-cell} ipython3
luminosities_lsun = jnp.array([0.08, 0.35, 1.0, 1.9, 4.5])
teffs_k = jnp.array([3800.0, 4800.0, 5780.0, 6300.0, 6900.0])

inner_au, outer_au = jax.vmap(habitable_zone_au)(luminosities_lsun)
k_inner_au, k_outer_au = jax.vmap(kopparapu_habitable_zone_au)(
    luminosities_lsun, teffs_k
)
for row in zip(luminosities_lsun, teffs_k, inner_au, outer_au, k_inner_au, k_outer_au):
    print("L = {:.2f}  Teff = {:.0f} K  EEID: {:.3f}-{:.3f} AU  "
          "Kopparapu: {:.3f}-{:.3f} AU".format(*map(float, row)))
```

## How the two definitions compare

In units of the EEID, the scaled edges are flat lines, while the
effective-flux limits move outward around cooler hosts, whose redder light
warms a planet more per unit of bolometric flux.

```{code-cell} ipython3
import hwostyle
import matplotlib.pyplot as plt

hwostyle.use("light")

teff_k = jnp.linspace(2600.0, 7200.0, 200)
fig, ax = plt.subplots(figsize=(6, 4.4), layout="constrained")
for optimistic, style in ((False, "-"), (True, "--")):
    label = "optimistic" if optimistic else "conservative"
    inner_eeid, outer_eeid = kopparapu_habitable_zone_au(1.0, teff_k, optimistic)
    ax.plot(teff_k, inner_eeid, "C0" + style, label=f"Kopparapu {label}")
    ax.plot(teff_k, outer_eeid, "C0" + style)
    flat_inner, flat_outer = habitable_zone_au(1.0, optimistic)
    ax.axhline(float(flat_inner), color="C1", ls=style, label=f"EEID-scaled {label}")
    ax.axhline(float(flat_outer), color="C1", ls=style)
ax.set_xlabel("Host effective temperature [K]")
ax.set_ylabel("Edge distance [EEID]")
fig.legend(loc="outside lower center", ncols=2, fontsize="small")
plt.show()
```

## References

- Kopparapu, R. K., et al. 2013, ApJ, 765, 131
- Kopparapu, R. K., et al. 2014, ApJL, 787, L29
- Stark, C. C., et al. 2019, JATIS, 5, 024009
