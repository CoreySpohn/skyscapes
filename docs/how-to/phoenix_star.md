# Build a star from a PHOENIX model atmosphere

{func}`skyscapes.io.from_phoenix` returns a {class}`~skyscapes.scene.Star` whose
spectrum is a PHOENIX BT-Settl model atmosphere chosen by effective
temperature and surface gravity. The models come from the solar-metallicity
(`phoenixm00`) grid of the STScI reference atlases, which covers 1.3 nm to
77.6 micron, so it reaches the 3.9 to 5.0 micron window where the CO
absorption of M dwarfs makes a blackbody wrong in shape as well as amplitude.

## Grid and cache

Each Teff node, 2000 K to 12000 K, is one 11.5 MB file holding every
tabulated `logg`. All 76 nodes are registered with their SHA-256 hashes in
{mod}`skyscapes.datasets`. A node is downloaded from archive.stsci.edu the
first time a model needs it, checked against its hash, and cached in the
`skyscapes` directory of the OS cache; a call needs one file at a node Teff
and two between nodes.

```python
from skyscapes.datasets import list_phoenix

nodes = list_phoenix()
assert nodes[0] == 2000 and nodes[-1] == 12000 and 3500 in nodes
```

## A star at a distance

The surface flux is diluted by `(radius_m / distance)**2`, the same
normalization as `BlackbodyStar`. Only solar metallicity is registered, so
`mh` must be 0.

```python notest
# downloads two 11.5 MB files from archive.stsci.edu on first use
import jax.numpy as jnp
from hwoutils.constants import Msun2kg, Rsun2m

from skyscapes.io import from_phoenix

star = from_phoenix(
    3550.0,
    5.0,
    Ms_kg=0.4 * Msun2kg,
    dist_pc=5.0,
    radius_m=0.4 * Rsun2m,
    wavelength_range_nm=(3000.0, 6000.0),
)
wavelength_nm = jnp.linspace(3900.0, 5000.0, 200)
photon_flux = star.spec_flux_density(wavelength_nm, jnp.zeros(()))  # ph/s/m^2/nm
```

`phoenix_surface_flux(teff_K, logg)` returns the model surface flux in
W/m^2/nm on the native wavelength grid, for normalizing to a magnitude or a
luminosity instead of a radius and distance.

## Interpolation

Between nodes the loader interpolates log flux linearly in Teff and in `logg`,
and a parameter that sits on a node returns that node unchanged. Linear
interpolation in flux itself would lift the midpoint of a 100 K step by about
3 percent at 100 to 300 nm and under 0.2 percent in the infrared, because the
ultraviolet flux changes exponentially with temperature. A `logg` outside the
range tabulated for the node file (the range is narrower for hot and for cool
nodes) raises `ValueError`.
