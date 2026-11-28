# Changelog

## [6.0.0](https://github.com/CoreySpohn/skyscapes/compare/v5.1.0...v6.0.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **disk:** make GraterDisk fields after the geometric core keyword-only
* **skyscapes:** close scene, disk, spectral and population forwards at their working precision
* **skyscapes:** keep ExoVista cadences relative before narrowing, decide annulus support on one float32 rule, refuse float32 inference densities
* **io:** from_exovista no longer reads header A and E; each selected planet reproduces its file position and velocity relative to the star at the first epoch, and multi-planet scenes move accordingly. Adds Scene.from_cartesian.
* **disk:** parametric disks (GraterDisk, ExovistaParametricDisk) no longer have the inclination_deg and position_angle_deg leaves; photon_radiance and photon_radiance_at on every disk require inclination_deg and position_angle_deg keyword arguments (stored cubes and AYOExozodi accept and ignore them); orient_disk is removed; Scene.midplane_inc_deg and Scene.midplane_position_angle_deg are float array leaves and are passed to every disk read; plot_disk_radiance requires incl_deg and position_angle_deg for a bare disk. Move the angles from the disk to the scene, or pass them to the read.
* **scene:** Star.metadata and the metadata keyword are removed and the loaders no longer write it; read the ExoVista star header with skyscapes.io.exovista_header(path), which returns a plain dictionary.
* **scene:** a scene built with an inactive slot of nonzero mass now gives the valid planets the orbit rates of the star and valid planets alone; earlier versions added the inactive mass to the enclosed mass.
* **scene:** Scene.from_scenes, to_scenes and from_table default to device=None (uncommitted arrays on the JAX default device); pass device="cpu" to commit a batch to the CPU.
* **scene:** Scene.from_scenes, to_scenes and from_table return leaves that are jax.Array in the active dtype (not NumPy float64) and take device= (default "cpu"); the source layout is an int32 array with -1 as the sentinel; to_table writes active-dtype values.
* **tables:** ProductTable holds its cell edges as a HostGrid without a tolerance merge, so product knot digests change.
* **tables:** Star, PrecomputedZodi, GridPhysicalModel (field renamed wavelengths_nm to knots), TimeSeriesPhysicalModel, ExovistaDisk and the parametric disks hold their wavelength grid as static host float64 data, and the ExoJAX and cached models hold nu_grid as a static HostGrid; knots are no longer leaves, so eqx.tree_at cannot reach them (build a new source). knot_digest is a property, Scene.knot_leaves and table_digest are removed, a source built from traced knots raises TypeError, ExoVista loaders widen stored microns to float64 before scaling (knot digests change), and Scene.from_scenes refuses two star grids that differ, naming twins that differ only by rounding.
* **scene:** Scene.band_contrasts takes a required keyword resolution (Resolution.build(quadrature, *scene.spectral_tables()), or Resolution() for a scene with no table) and no longer samples sources at the nodes; a band that leaves a source table is refused at Resolution.build instead of returning NaN in an open wing.
* **disk:** density_level under luminosity_rule="bolometric" now averages the host star over the calibration band (a 506 to 594 nm top hat for Johnson V) through star.calibration_flux_density instead of reading it at the passband center, so a bolometric level moves by the line error and a tabulated Star refuses: compute the calibration through Scene.sources with one Quadrature.build((band,)) per band of Scene.calibration_bands() in the Resolution, and render with Sources.node_star. A non-V passband under the bolometric rule needs an explicit calibration_band. The band rule no longer reads the star.
* **disk:** keep the cube and the star table as the live leaves, derive the spline on use
* **scene:** flux_scale leaf on every emitter, public flux methods call private hooks
* **io:** rename bandpass_from_table to transmission_from_table
* **scene:** rename the band_contrasts bandpass argument to transmission
* **scene:** host the local-zodi models, take the sky frame and time in photon_radiance and merge AYOExozodi into the disk

### Features

* **disk:** a surface-brightness disk calibrates over a band, averaging the host star under the bolometric rule and not reading it under the band rule ([a567fff](https://github.com/CoreySpohn/skyscapes/commit/a567fff9740400529eed894f38047cc4374c8b11))
* **disk:** keep the cube and the star table as the live leaves, derive the spline on use ([cfa8f65](https://github.com/CoreySpohn/skyscapes/commit/cfa8f653f251b1ebca85e091b8942d07f14acff6))
* **disk:** make GraterDisk fields after the geometric core keyword-only ([ec32d8f](https://github.com/CoreySpohn/skyscapes/commit/ec32d8fb6e1fc297c962abd42ee85c013c4960db))
* **disk:** the scene owns the disk orientation and disk reads take it as arguments ([fbce272](https://github.com/CoreySpohn/skyscapes/commit/fbce2728fa5862506babd382594048e5d607d6e5))
* **io:** rename bandpass_from_table to transmission_from_table ([bdf8d00](https://github.com/CoreySpohn/skyscapes/commit/bdf8d00e6a164a06b4383aa71249ae443a90eb47))
* **scene:** batch results hold jax arrays and take device=; remove the host float64 packing ([87068ce](https://github.com/CoreySpohn/skyscapes/commit/87068cea885562091f7d1bd93000b8d9b98c5833))
* **scene:** default the batch device to the JAX default device ([a293743](https://github.com/CoreySpohn/skyscapes/commit/a2937438426ff0ca188102df763086781998c052))
* **scene:** flux_scale leaf on every emitter, public flux methods call private hooks ([bc8a184](https://github.com/CoreySpohn/skyscapes/commit/bc8a18496d6b6ef19426addbcdad540cce908dad))
* **scene:** host the local-zodi models, take the sky frame and time in photon_radiance and merge AYOExozodi into the disk ([88c2a6f](https://github.com/CoreySpohn/skyscapes/commit/88c2a6f1f7e6220da1667e71b184b6bd6b86c9f5))
* **scene:** name the knot leaves of a scene's tables with Scene.knot_leaves ([185c172](https://github.com/CoreySpohn/skyscapes/commit/185c172822e6c5721b1a15b76ca7a4d743cb49ca))
* **scene:** remove Star.metadata and add io.exovista_header ([cdb69c5](https://github.com/CoreySpohn/skyscapes/commit/cdb69c5d9d40a1388d67d6032fe6fde7b92001a7))
* **scene:** require a resolution in Scene.band_contrasts ([e119f9d](https://github.com/CoreySpohn/skyscapes/commit/e119f9d226ff71cd4bc59573e1a17bb67cfeceb6))
* **scene:** Scene.sources reads every source once on a band rule, with spectral_tables listing the tables a Resolution needs ([a9c8d26](https://github.com/CoreySpohn/skyscapes/commit/a9c8d26be825cbe49ca3a5dd507b8698f23ffec6))
* **scene:** stars, tabulated planet models and PrecomputedZodi carry knot digests and read node-effective values on a band rule ([d848c3b](https://github.com/CoreySpohn/skyscapes/commit/d848c3b5c123d0a8703dd530ab4df28892127c99))
* **tables:** hold every table's knots as a static Knots and derive its digest ([16c42f1](https://github.com/CoreySpohn/skyscapes/commit/16c42f158c9c3f1fc17ca5cade1e53af56f3a565))
* **tables:** integrate a planet times its star on the raw union of knots, bound the grid caches, and name twin planet grids and one-wavelength tables ([b5c2d65](https://github.com/CoreySpohn/skyscapes/commit/b5c2d65c1aaa98f568339847ecdb9725feae52aa))


### Bug Fixes

* import the solar gravitational parameter from hwoutils and rotate helpers from orbix.system ([14b22f7](https://github.com/CoreySpohn/skyscapes/commit/14b22f77b91cec05a4416fe0196308c94c914e81))
* **io:** build ExoVista planets from first-epoch states in a Jacobi chain ([d89a43a](https://github.com/CoreySpohn/skyscapes/commit/d89a43a8c92830e467c6cbd56e5bb1a3272da6df))
* **io:** keep ExoVista phase-mode spectra for face-on and near-face-on planets ([2744673](https://github.com/CoreySpohn/skyscapes/commit/27446739059154f5e60551251fedd7618123857b))
* **physical_model:** divide a blackbody-star thermal contrast by the star's flux scale ([28dc3e9](https://github.com/CoreySpohn/skyscapes/commit/28dc3e9ea7657936502ce80785b415f9ddffbdef))
* **population:** broadcast scalar star leaves of HostStars to the hosts axis ([ddbd257](https://github.com/CoreySpohn/skyscapes/commit/ddbd2576ba29ce7b39a26d6036a415a5350ea0f9))
* **scene:** check a concrete flux_scale with NumPy so node_star builds under jit ([50112a0](https://github.com/CoreySpohn/skyscapes/commit/50112a0d5331774780b3c27c1f7699b7c8e5968f))
* **scene:** name NumPy knot grids in Scene.knot_leaves ([d3fcfc7](https://github.com/CoreySpohn/skyscapes/commit/d3fcfc7b56014adcadf8adad9e645f589485f2e6))
* **scene:** raise a table's own knot error in batches, build product edges as ProductEdges, and bound the knot caches at 32 ([9785cbf](https://github.com/CoreySpohn/skyscapes/commit/9785cbfc2b2800d93f0bc1bc053e0cc8ea89802f))
* **scene:** rebuild scenes through one helper and mask inactive masses first ([e7285e6](https://github.com/CoreySpohn/skyscapes/commit/e7285e6c25b666eca20418aa7fa699d0bc4f45c0))
* **scene:** refuse non-finite and out-of-range table layouts; byte-level bitwise test helper ([90c4056](https://github.com/CoreySpohn/skyscapes/commit/90c4056e1c8d17bcd4aae65776e25abc8982b291))
* **skyscapes:** accept a scalar observation time in Scene.sources again ([87fddb3](https://github.com/CoreySpohn/skyscapes/commit/87fddb3e3cd509a80eaac08dcdb66fce76b142a7))
* **skyscapes:** close scene, disk, spectral and population forwards at their working precision ([aa4f792](https://github.com/CoreySpohn/skyscapes/commit/aa4f7924e07cb324cc2c7877bc794e527d0f6c16))
* **skyscapes:** keep ExoVista cadences relative before narrowing, decide annulus support on one float32 rule, refuse float32 inference densities ([c41c18e](https://github.com/CoreySpohn/skyscapes/commit/c41c18e9a76ae138a20808425aad90e13f3a9f1a))
* **skyscapes:** keep the Lambert crescent near new phase with a series in the distance from pi ([a27b508](https://github.com/CoreySpohn/skyscapes/commit/a27b508ccdbac925de6a4327c2528ab3b9964501))
* **skyscapes:** read zodi, albedo and opacity tables with interp_linear for jax 0.12.1 ([2eedd3b](https://github.com/CoreySpohn/skyscapes/commit/2eedd3b4c810b21a769c94584e28f411948cb5e3))


### Code Refactoring

* **scene:** rename the band_contrasts bandpass argument to transmission ([36e2876](https://github.com/CoreySpohn/skyscapes/commit/36e2876eb22435a909e486783680602fb1fa8d1d))

## [5.1.0](https://github.com/CoreySpohn/skyscapes/compare/v5.0.0...v5.1.0) (2026-11-20)


### Miscellaneous Chores

* release 5.1.0 ([703f790](https://github.com/CoreySpohn/skyscapes/commit/703f7903dd1e46fe59539436e21fe72fd7b5b88d))

## [5.0.0](https://github.com/CoreySpohn/skyscapes/compare/v4.1.0...v5.0.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* declare disk radiance as a per-steradian density
* declare spectral breaks on every scene model and read band contrasts from one quadrature
* rename dist_AU to star_planet_distance_AU across physical models and scene
* one reflected-light projection, public reflectivity(), star_planet_distance_AU keyword
* put angular diameter on the stellar interface and query disk radiance at sky positions
* read band transmissions, conversions and colours from hwoutils and eyepiece
* Planck radiance and band centres come from hwoutils
* local-zodi models come from zodi.jax and skyscapes.background is gone
* one Lambert phase and reflected contrast, orbit mechanics from orbix, Legendre-only disk sightlines and no opacity alias
* disks return a photon radiance image on a caller grid from a calibrated dust amplitude, in astronomical position angle
* stars, planets and local zodi return photon quantities as functions of wavelength

### Features

* declare spectral breaks on every scene model and read band contrasts from one quadrature ([a024600](https://github.com/CoreySpohn/skyscapes/commit/a02460082c73eb8d5088b1e648b3ec68b8702e7d))
* disks return a photon radiance image on a caller grid from a calibrated dust amplitude, in astronomical position angle ([07ef2a3](https://github.com/CoreySpohn/skyscapes/commit/07ef2a368ee6412918bf72f3c8893472e6d65676))
* **population:** exoEarth-candidate flag for the planets of a scene ([ac988b1](https://github.com/CoreySpohn/skyscapes/commit/ac988b10b093b001ee1e62d9882373b2248835ee))
* **population:** read a one-planet mark to coordinates and rebuild the system from them ([b5cd835](https://github.com/CoreySpohn/skyscapes/commit/b5cd8354441ebf7795607867aad12e85a1b2bd11))
* put angular diameter on the stellar interface and query disk radiance at sky positions ([d9ba388](https://github.com/CoreySpohn/skyscapes/commit/d9ba388c42bf94f2742486d88ac2b005beec181a))
* stars, planets and local zodi return photon quantities as functions of wavelength ([c55c3b1](https://github.com/CoreySpohn/skyscapes/commit/c55c3b1976d9720514ab2ffcacc6416b8bdb3f04))
* **zodi:** declare the Table 19 colour breakpoints for band quadrature ([4c8b2ef](https://github.com/CoreySpohn/skyscapes/commit/4c8b2ef5573e6458f80e6f2a194b6f3f7813b888))


### Bug Fixes

* declare disk radiance as a per-steradian density ([141a067](https://github.com/CoreySpohn/skyscapes/commit/141a0678c94fb0c262a93f9295ee83989de51ad3))
* **physical_model:** form the blackbody contrast from log occupation factors, which keeps the star-temperature gradient under jit ([4724737](https://github.com/CoreySpohn/skyscapes/commit/47247375d5e941a5a551a9a2c7000fd03e425d7c))


### Code Refactoring

* local-zodi models come from zodi.jax and skyscapes.background is gone ([9703858](https://github.com/CoreySpohn/skyscapes/commit/9703858047193884caa6b236bbf309611160287c))
* one Lambert phase and reflected contrast, orbit mechanics from orbix, Legendre-only disk sightlines and no opacity alias ([823cf14](https://github.com/CoreySpohn/skyscapes/commit/823cf14f5db4ccb401992d37626a86b9c0553a00))
* one reflected-light projection, public reflectivity(), star_planet_distance_AU keyword ([adb3fe7](https://github.com/CoreySpohn/skyscapes/commit/adb3fe742d6ea2abdb85e874671b5dfd790b5142))
* Planck radiance and band centres come from hwoutils ([e590fe6](https://github.com/CoreySpohn/skyscapes/commit/e590fe621b62d415608892efe77b2faccd86f34c))
* read band transmissions, conversions and colours from hwoutils and eyepiece ([aa1cddc](https://github.com/CoreySpohn/skyscapes/commit/aa1cddce7a5c38e5f16cd3a3d7e2a489a05d9f40))
* rename dist_AU to star_planet_distance_AU across physical models and scene ([52cc018](https://github.com/CoreySpohn/skyscapes/commit/52cc018528bb5de0baa08307a63067762a74d847))

## [4.1.0](https://github.com/CoreySpohn/skyscapes/compare/v4.0.1...v4.1.0) (2026-11-16)


### Features

* **scene:** a padded, masked batch of systems on the scene System, with table and list round trips ([15c2c74](https://github.com/CoreySpohn/skyscapes/commit/15c2c745112e49c31541ec0b1ebb679d5510744c))

## [4.0.1](https://github.com/CoreySpohn/skyscapes/compare/v4.0.0...v4.0.1) (2026-11-16)


### Bug Fixes

* the docs example imports its names and the floor test expects orbix 4.1.0 ([8449ac0](https://github.com/CoreySpohn/skyscapes/commit/8449ac0389dfe32ffeaea36376c05e524224abb0))

## [4.0.0](https://github.com/CoreySpohn/skyscapes/compare/v3.0.0...v4.0.0) (2026-11-16)


### ⚠ BREAKING CHANGES

* **scene:** times are days from the scene epoch
* **scene:** the scene holds an orbix system; stars and planets are photometric only

### Features

* **io:** the ExoVista loader accepts a shared scene epoch ([d6137c0](https://github.com/CoreySpohn/skyscapes/commit/d6137c0fce24487be64596eec4da6c6d0727f012))
* position angles, stellar reflex and radial velocity on System, and a Bandpass with band contrasts ([9fa7a56](https://github.com/CoreySpohn/skyscapes/commit/9fa7a56d666a5f7c21e728f049c9541eb21e2be6))
* scene methods read one solved Jacobi ephemeris from an orbix system and accept a shared ephemeris ([883dabe](https://github.com/CoreySpohn/skyscapes/commit/883dabe6f16eeac915ec713d60df292c41edf355))
* **scene:** the scene holds an orbix system; stars and planets are photometric only ([bdcaf5c](https://github.com/CoreySpohn/skyscapes/commit/bdcaf5cdc79c3878abd7ac7b0a8553b0cadaf417))
* **scene:** times are days from the scene epoch ([5ecb6b5](https://github.com/CoreySpohn/skyscapes/commit/5ecb6b53820ed06c4c9a14ffc7cb3727a58d2891))

## [3.0.0](https://github.com/CoreySpohn/skyscapes/compare/v2.2.0...v3.0.0) (2026-11-15)


### ⚠ BREAKING CHANGES

* unit suffixes keep the unit's capitalization and time arguments are time_jd

### Features

* **disk:** finite line-of-sight window through edge-on and optional smooth radial cutoffs ([8f968cc](https://github.com/CoreySpohn/skyscapes/commit/8f968cc1163a9f832ed1ff69726b4bf1a36ae199))
* **disk:** Gauss-Legendre (default), Golub-Welsch Gauss and trapezoid line-of-sight rules, and a fixed vertical profile fast path ([79fd3ac](https://github.com/CoreySpohn/skyscapes/commit/79fd3ac2193ffc3235b40b62462887d8b563c3ad))
* **io:** Castelli-Kurucz ATLAS9 stellar grid loader for stars above 12000 K ([50d37a0](https://github.com/CoreySpohn/skyscapes/commit/50d37a0bcddaa9cbaac6461d335c57b877b06d09))
* **physical_model:** blackbody thermal emission against the host star, and System.contrast_cube ([0e23324](https://github.com/CoreySpohn/skyscapes/commit/0e23324b37a487535b197c2cd9351761f3be202c))
* **physical_model:** planet flux is the primary quantity; a sum model carries reflected and thermal light ([5fba655](https://github.com/CoreySpohn/skyscapes/commit/5fba655e31e0b0e942794c29ea01646a33d24426))
* **scene:** bin-averaged stellar flux and flux-weighted bin contrast ([31d0656](https://github.com/CoreySpohn/skyscapes/commit/31d06562fecec922da60ca843004d64edf21ca23))
* **scene:** expose a tabulated star's wavelength grid and range as read-only properties ([d4383a8](https://github.com/CoreySpohn/skyscapes/commit/d4383a87b286b8f565ec0300719b41a920e54be5))
* **spectral:** one band-mean function for star, contrast and planet flux, with cost set by the knots inside each bin ([3a187f5](https://github.com/CoreySpohn/skyscapes/commit/3a187f58e975ea6d86bc1e538e1ec12c6381c5b5))


### Bug Fixes

* **background:** remove the extra wavelength factor from the LeinertZodi photon flux ([7f7bb43](https://github.com/CoreySpohn/skyscapes/commit/7f7bb435795ffcbc99397b16589d4e753d845625))
* **blackbody:** balance Planck prefactors so single-precision second derivatives hold under JIT ([182b9fb](https://github.com/CoreySpohn/skyscapes/commit/182b9fb073fbbe1a3d9c83e3a4b2a22c0fca5090))
* **disk:** include the ring edges in the ExoVista ring branch so the density is continuous there ([877f9f5](https://github.com/CoreySpohn/skyscapes/commit/877f9f5304299a9f2e2094411764033cad7ab639))
* **population:** clip negative planet-type albedos on load and make radius_to_mass accept JAX arrays ([43b1dc3](https://github.com/CoreySpohn/skyscapes/commit/43b1dc3f92f92221d47d88e5108ec21f67d5f6ed))
* **scene:** accept a one-element wavelength array in contrast ([2e4c6eb](https://github.com/CoreySpohn/skyscapes/commit/2e4c6eb7e78e90c0ebf76f56bdb19799561206af))
* **scene:** Planck terms grouped so temperature derivatives are finite in single precision ([5c62a72](https://github.com/CoreySpohn/skyscapes/commit/5c62a72a064feea28dedbcade9cdbb71ddb6e83c))
* **scene:** reject wavelength arrays in contrast, pass wavelength through alpha_dMag, share one Planck function ([2b8d4ff](https://github.com/CoreySpohn/skyscapes/commit/2b8d4ff7c104d7ae7c5fb3cce511488fd8404fbf))


### Code Refactoring

* unit suffixes keep the unit's capitalization and time arguments are time_jd ([39dfffb](https://github.com/CoreySpohn/skyscapes/commit/39dfffb30976d0a87670f6134ca01110fb9bcd0a))

## [2.2.0](https://github.com/CoreySpohn/skyscapes/compare/v2.1.0...v2.2.0) (2026-11-12)


### Features

* **io:** add PHOENIX model-atmosphere loader from_phoenix with a hash-registered STScI grid ([6791b70](https://github.com/CoreySpohn/skyscapes/commit/6791b702c3416f632c44b4ceabb60fc9348a86f3))
* **scene:** add diameter_arcsec to FlatStar and BlackbodyStar ([bf22955](https://github.com/CoreySpohn/skyscapes/commit/bf229556edfde721a127afd335ca4277bf51a698))
* **skyscapes:** add occurrence-rate planet population draw ([83c500f](https://github.com/CoreySpohn/skyscapes/commit/83c500f6f9e66aab7d29ef75d5824284bb7a8689))


### Bug Fixes

* **skyscapes:** delegate zodi tables and color to zodi, fixing the AYO default color in AB ([591365d](https://github.com/CoreySpohn/skyscapes/commit/591365daf35500db4145f0992d3b190795ee7cc1))

## [2.1.0](https://github.com/CoreySpohn/skyscapes/compare/v2.0.0...v2.1.0) (2026-11-05)


### Features

* **exojax:** add ExoJaxEmissionModel for thermal-infrared planet spectra ([a50d4b8](https://github.com/CoreySpohn/skyscapes/commit/a50d4b8254dd73da86a60628b2bd959c66dc6b4c))
* **exojax:** add polynomial T-P profile, emitting surface and HITRAN-only species option ([88c1029](https://github.com/CoreySpohn/skyscapes/commit/88c1029694b50484d84122bb169bb41c56a78347))
* **exojax:** optional HITRAN collision-induced absorption in the emission model ([396baf0](https://github.com/CoreySpohn/skyscapes/commit/396baf0e32ae303eab6b1bd5614f1720f0622baa))
* **habitable_zone:** add EEID-scaled and Kopparapu habitable-zone edges ([80626f4](https://github.com/CoreySpohn/skyscapes/commit/80626f49403f46d6f84d77a3259cf9b4257f06c8))
* **scene:** add BlackbodyStar, a diluted Planck spectrum from Teff, radius and distance ([0d4eb07](https://github.com/CoreySpohn/skyscapes/commit/0d4eb07dc2907d9acadf877e294d00258428b902))


### Bug Fixes

* **disk:** render images in the FITS display convention with +column west ([8331be5](https://github.com/CoreySpohn/skyscapes/commit/8331be52c96a4af1b35896b7ca619bcf1066091c))
* **exojax:** end emission layers at the surface pressure boundary ([a0e1497](https://github.com/CoreySpohn/skyscapes/commit/a0e14979093f57c16d26a4f5621697fa30998800))
* **exojax:** size reflected grids by resolving power so line-resolved bands converge ([453aba5](https://github.com/CoreySpohn/skyscapes/commit/453aba50acd0829c9eaeb67e9a6fa5d448ef0ef6))
* **io:** check PHOENIX conversion inputs and warn when VMAG is unusable ([b251d62](https://github.com/CoreySpohn/skyscapes/commit/b251d62f1e6a945508d84c30bac38ddc6e4ac1ab))
* **io:** convert ExoVista (west, north, toward) states to the orbital-element frame at load ([d608a0e](https://github.com/CoreySpohn/skyscapes/commit/d608a0e1cbd55e0bbb89a64b2f927a307a050560))
* **io:** convert the ExoVista midplane position angle into the east/north chart ([51c19bc](https://github.com/CoreySpohn/skyscapes/commit/51c19bc1efef09f783d5849714603d4eba96763e))
* **io:** convert unmarked ExoVista 2.5+ stellar spectra from PHOENIX surface flux to observer Jy ([abd038c](https://github.com/CoreySpohn/skyscapes/commit/abd038cabb9b45b10afd8b45d4467eff08372047))
* **io:** derive the ExoVista stellar diameter from RSTAR and DIST, since system files store ANGDIAM as a radius ([3a29ec8](https://github.com/CoreySpohn/skyscapes/commit/3a29ec88a86fbc31490450df951b9a76f690283c))
* **io:** make ExoVista planet states star-relative before deriving orbital elements ([54a44af](https://github.com/CoreySpohn/skyscapes/commit/54a44afca31a220843f99a53c3a6ac4622af95a2))
* **io:** mirror ExoVista disk cubes at load so +column is east, matching the planets ([f175991](https://github.com/CoreySpohn/skyscapes/commit/f17599116ed4ce6290b106b20d14e4687cb567e6))
* **io:** require RSTAR and DIST for the ExoVista stellar diameter ([02e36c0](https://github.com/CoreySpohn/skyscapes/commit/02e36c06503cf4f7a2f1431d805a4ba2a24d3f44))
* **scene:** pass orbix (east, north, away) states through and convert ExoVista states by a proper rotation ([a06f896](https://github.com/CoreySpohn/skyscapes/commit/a06f896d5db887218eac1113dc5aaf40a0c8c0e2))
* **scene:** propagate planets with G (M_star + M_planet), matching the ExoVista load ([4159e2f](https://github.com/CoreySpohn/skyscapes/commit/4159e2f80fa156fe4a2e8292e5cca483e39e566a))
* **scene:** return planet positions in the (east, north, toward) sky chart with the standard phase angle ([fd028b2](https://github.com/CoreySpohn/skyscapes/commit/fd028b275802cd0ed082d44d3704304a857d0ce6))
* **skyscapes:** close HITRAN dataframes after line extraction ([12774fe](https://github.com/CoreySpohn/skyscapes/commit/12774fe849883624e4f5016d056185ad10d73e8d))
* **skyscapes:** keep disk cube and wavelength dtypes consistent ([74ef22a](https://github.com/CoreySpohn/skyscapes/commit/74ef22ae3945ee170590b552484304c66cac864b))
* **skyscapes:** read x64 configuration through the public API ([adc5456](https://github.com/CoreySpohn/skyscapes/commit/adc54563531fd2f91b58b867393e54fffbda9130))
* **skyscapes:** support single-epoch ExoVista phase models ([b42b230](https://github.com/CoreySpohn/skyscapes/commit/b42b2304277cafc4d54fcc5365ca8ea79f0d2f97))
* **viz:** label the illumination angle beta and scope the changelog history claims ([3782e86](https://github.com/CoreySpohn/skyscapes/commit/3782e86451814838ae5a86e65d71db0e4617bc83))

### Migration notes

* **disk, viz:** Images follow the FITS display convention: +column west,
  north up and east left with `origin="lower"`. Parametric disk images are
  mirrored left-right compared with earlier releases. ExoVista cubes are
  used as written, unchanged from 2.0.0. `plot_disk_image` no longer inverts
  the x axis separately; its descending extent sets the orientation.

* **io:** ExoVista scenes now match the true sky. In 2.0.0, planet RA offsets
  were labelled east but were west-positive; their east offsets now change
  sign. The cube was already used as written (+column west, east left), but
  `plot_disk_image` inverted the x axis assuming +column east; its displayed
  orientation now changes. `System.midplane_pa_deg` becomes (-PA) mod 360,
  so parametric disks and outlines built from `System.midplane_*` match the
  loaded image. Code placing planets with the old sign must be updated.

* **io:** ExoVista planet states are now made star-relative before conversion
  to orbital elements. Earlier releases used barycentric states, which biased
  the elements of massive planets.

* **io:** ExoVista planet positions and velocities are now converted from
  ExoVista's (west, north, toward) axes to (east, north, away) at load
  with the proper rotation (-x, y, -z). Earlier releases kept ExoVista's axes, so
  loaded elements (Omega, i, omega) followed a different convention, though
  the raw FITS state values round-tripped. Time-indexed contrast light
  curves are preserved. Phase-indexed contrasts are regridded on the
  star-relative phase, so they can differ slightly from earlier releases.

* **scene:** `Planet.position_arcsec` now returns (Delta alpha*, Delta Dec) =
  (east, north), where Delta alpha* = Delta RA cos Dec. `Planet.propagate`
  returns orbix's (east, north, away) positions unchanged: right-handed,
  z positive away and dz/dt positive receding, with the standard phase
  cos(beta) = z/r. This requires orbix>=3.0.0. Earlier releases returned
  raw orbix rows (north, east, away) re-labelled as east/north, swapping
  sky axes for planets built from orbital elements;
  ExoVista planets instead had west-positive offsets. In 1.6.1 and 2.0.0,
  element-built planets also had inverted phases (crescent and gibbous exchanged), changing
  `alpha_dMag`, `contrast` and `spec_flux_density`. ExoVista phase grids use
  the same standard angle at construction and evaluation.

* **viz:** Side views in `plot_system` and `plot_disk_geometry` place the
  observer on the left at -z, with z labelled away from the observer. The
  scattering diagram labels the illumination angle beta.

## [2.0.0](https://github.com/CoreySpohn/skyscapes/compare/v1.6.1...v2.0.0) (2026-10-13)


### ⚠ BREAKING CHANGES

* **io:** from_exovista raises ValueError unless jax_enable_x64 is set, and planet_spectra="auto" loads ExoVista 2.5+ scenes with time-indexed contrasts that return NaN outside the recorded epochs.

### Features

* **io:** time-series planet contrast, star spot records and ExoVista 2.5+ layouts ([946312a](https://github.com/CoreySpohn/skyscapes/commit/946312a8b06a8a0735fdd6f92511f7a1569f3e62))

## [1.6.1](https://github.com/CoreySpohn/skyscapes/compare/v1.6.0...v1.6.1) (2026-10-12)


### Bug Fixes

* **scene:** convert orbix's observer-axis angle to the standard phase angle ([65f7a01](https://github.com/CoreySpohn/skyscapes/commit/65f7a010426325f21975e089374a64597408440d))

## [1.6.0](https://github.com/CoreySpohn/skyscapes/compare/v1.5.0...v1.6.0) (2026-09-29)


### Features

* **viz:** add grain, observer, look-angle and disk-image view options ([3af8ded](https://github.com/CoreySpohn/skyscapes/commit/3af8dedf9af9edc34ce3b7998b092297214d8215))


### Bug Fixes

* **viz:** keep the top view's longitude label clear of the rays ([50304c1](https://github.com/CoreySpohn/skyscapes/commit/50304c167f18c95006ad5fc9e540250a923d5e40))
* **viz:** set degree signs tight against the number in every view ([d9caca7](https://github.com/CoreySpohn/skyscapes/commit/d9caca7f49305f5f3ab17ed00eef571fa5958f79))

## [1.5.0](https://github.com/CoreySpohn/skyscapes/compare/v1.4.0...v1.5.0) (2026-09-29)


### Features

* **viz:** add plot_scattering_angle, the grain inset's construction at full size ([74f7636](https://github.com/CoreySpohn/skyscapes/commit/74f7636272af369eea922c1eaead5b2a01013117))
* **viz:** add show= parts to the disk and local-zodi geometry views and several grains per zodi sightline ([e55f25e](https://github.com/CoreySpohn/skyscapes/commit/e55f25e2f473b5b34ed946d36a277e18a10646dd))
* **viz:** add system, disk geometry and local zodi views ([572c298](https://github.com/CoreySpohn/skyscapes/commit/572c298937d942f13c5b99423b544137d33315fa))


### Bug Fixes

* **disk:** keep LOS path weights positive above 90 deg, reject edge-on in the shared kernel, and add node and support convergence ladders ([c07c435](https://github.com/CoreySpohn/skyscapes/commit/c07c435ff4558103d967a087b4a0c0e8f323a23e))
* **viz:** refuse negated disk maps, unify inclination range, and keep labels and insets clear ([2907a0c](https://github.com/CoreySpohn/skyscapes/commit/2907a0cb4e8c94e2ad933b961b3bc575431bfd28))
* **viz:** shorten the local-zodi side view's x label so it fits in the DejaVu fallback font used on CI ([cf325ea](https://github.com/CoreySpohn/skyscapes/commit/cf325eadbf2a9b7a8f1b8acf07be5e440dbca88e))
* **viz:** wrap the local-zodi side view's x label so it fits the figure ([cc29c47](https://github.com/CoreySpohn/skyscapes/commit/cc29c47940e1b9e9938721ce566c5a34e90a9e86))

## [1.4.0](https://github.com/CoreySpohn/skyscapes/compare/v1.3.0...v1.4.0) (2026-07-18)


### Features

* **background:** absorb Leinert/AYO zodi physics from orbix; clamp near-Sun domain ([71b7ad8](https://github.com/CoreySpohn/skyscapes/commit/71b7ad8175f063dcb48dfcc5cec668914322aec1))
* **io:** add load_disk flag to from_exovista ([a6ec2a6](https://github.com/CoreySpohn/skyscapes/commit/a6ec2a6a8bdfe366828eb219e050ebfe2cd04459))

## [1.3.0](https://github.com/CoreySpohn/skyscapes/compare/v1.2.0...v1.3.0) (2026-06-23)


### Miscellaneous Chores

* release 1.3.0 ([36758fd](https://github.com/CoreySpohn/skyscapes/commit/36758fd0fbf2ae915697c0347b82024339c0ac07))

## [1.2.0](https://github.com/CoreySpohn/skyscapes/compare/v1.1.1...v1.2.0) (2026-06-23)


### Features

* Add zenodo info ([9ff1e42](https://github.com/CoreySpohn/skyscapes/commit/9ff1e42be602462dc4a1e450ca3102c876ddf784))

## [1.1.1](https://github.com/CoreySpohn/skyscapes/compare/v1.1.0...v1.1.1) (2026-06-23)


### Bug Fixes

* **io:** drop double sky-rotation of ExoVista planet states so planets load coplanar with their disk ([1ab4582](https://github.com/CoreySpohn/skyscapes/commit/1ab4582c88f32f56c94d0497bccfdb74a4ca5092))

## [1.1.0](https://github.com/CoreySpohn/skyscapes/compare/v1.0.3...v1.1.0) (2026-06-20)


### Features

* **physical_model:** ExoJaxPhysicalModel.for_retrieval (precomputed-opacity factory) ([d408a82](https://github.com/CoreySpohn/skyscapes/commit/d408a82c1e0a909022f1a2b93cbc476eb478d6c3))
* **physical_model:** PrecomputedAbsorption (fixed-TP opacity, live abundance) ([bb4c6a0](https://github.com/CoreySpohn/skyscapes/commit/bb4c6a08f3864a0be6ec1a8a205d182eaea8f8ce))


### Bug Fixes

* Add guard to avoid x86 error ([a50b3d8](https://github.com/CoreySpohn/skyscapes/commit/a50b3d80170bc85ca99c77a218ffb6a151cf8be6))

## [1.0.3](https://github.com/CoreySpohn/skyscapes/compare/v1.0.2...v1.0.3) (2026-06-19)


### Bug Fixes

* clamp ExoVista contrast grid to non-negative on load ([bd549c2](https://github.com/CoreySpohn/skyscapes/commit/bd549c293e05a8f0957789db98718cdbb120fe53))
* **disk:** log-spaced LOS quadrature, relax GraterDisk edge-on guard to true singularity ([b613c75](https://github.com/CoreySpohn/skyscapes/commit/b613c75d96c5113ba02a3050350b56a4d40f5800))

## [1.0.2](https://github.com/CoreySpohn/skyscapes/compare/v1.0.1...v1.0.2) (2026-05-25)


### Bug Fixes

* Update docs and readme to compile properly ([73a19d2](https://github.com/CoreySpohn/skyscapes/commit/73a19d2f4d72f2a6e4d9945f428ef8473ad93286))

## [1.0.1](https://github.com/CoreySpohn/skyscapes/compare/v1.0.0...v1.0.1) (2026-05-25)


### Bug Fixes

* Add mermaid chart rendering to the docs ([3f4a8f7](https://github.com/CoreySpohn/skyscapes/commit/3f4a8f731c56a113318dd447625469daac1cb2f8))

## [1.0.0](https://github.com/CoreySpohn/skyscapes/compare/v0.4.0...v1.0.0) (2026-05-25)


### ⚠ BREAKING CHANGES

* Atmosphere models, docs, refactor language, and loaders
* SpectrumStar -> Star (canonical type gets the bare noun; previous Star/Spectrum compound was awkward); SimpleStar -> FlatStar (adjective-first variant naming, accurately describes the flat-spectrum behaviour); ZodiSourceAYO -> AYOZodi, ZodiSourceLeinert -> LeinertZodi, ZodiSourcePhotonFlux -> PrecomputedZodi (adjective-first prefix; Source suffix was redundant since the classes live in skyscapes.background and PrecomputedZodi now parallels PrecomputedReflectivity for tabulated-data variants); ZodiSource union -> Zodi.

### Features

* Atmosphere models, docs, refactor language, and loaders ([09e68ca](https://github.com/CoreySpohn/skyscapes/commit/09e68ca4b97b6378eee22528e45767b5702bc6e7))
* rename Star/Zodi classes for naming consistency ([eeba5e3](https://github.com/CoreySpohn/skyscapes/commit/eeba5e313536b0ab5873810d2dca36e63f37016a))

## [0.4.0](https://github.com/CoreySpohn/skyscapes/compare/v0.3.0...v0.4.0) (2026-05-22)


### Features

* Add _repr helpers and Scene/System/Planet/Star/Zodi reprs ([4a2f819](https://github.com/CoreySpohn/skyscapes/commit/4a2f81947c024bdfe89d2fb21bdf1eb2292cbc9b))
* **atmosphere:** Add ExoJax atmosphere and PrecomputedReflectivity cache ([426b520](https://github.com/CoreySpohn/skyscapes/commit/426b5203271a5abde67fc39b33040ee9812feb9c))
* **disk:** Add reprs to disk models ([0a91fe8](https://github.com/CoreySpohn/skyscapes/commit/0a91fe827e5d0511c68763e6f74fb1169f0b4713))


### Bug Fixes

* Enable x64 when running tests ([f970de2](https://github.com/CoreySpohn/skyscapes/commit/f970de2716c9271f7e470f714631c8df13354452))
* Pin vaex-core&gt;=4.19 for py3.12 wheels (radis transitive) ([9b7f830](https://github.com/CoreySpohn/skyscapes/commit/9b7f830574bc44fcfd5cdae9ea2b94140be0e67c))

## [0.3.0](https://github.com/CoreySpohn/skyscapes/compare/v0.2.0...v0.3.0) (2026-05-21)


### Features

* Add initial disk forward models ([d97e29c](https://github.com/CoreySpohn/skyscapes/commit/d97e29c5e511e71f557d00e912230812240a72b2))
* Exovista loading (coordinate system translation, ZodiSource) matches exoverses ([8b2462a](https://github.com/CoreySpohn/skyscapes/commit/8b2462ae20e921d34811cfc3ffcf0b55cad15219))

## [0.2.0](https://github.com/CoreySpohn/skyscapes/compare/v0.1.0...v0.2.0) (2026-05-14)


### Features

* Adding larger scene container and a more explicit approach to background sources support future expansion ([8399b74](https://github.com/CoreySpohn/skyscapes/commit/8399b7409a39d0874c3f8c9970e2232562311c65))

## [0.1.0](https://github.com/CoreySpohn/skyscapes/compare/v0.0.1...v0.1.0) (2026-04-23)


### Features

* **skyscapes:** add AbstractAtmosphere and LambertianAtmosphere ([28402af](https://github.com/CoreySpohn/skyscapes/commit/28402af54a185b365b000da9eff4f733541d434e))
* **skyscapes:** add GridAtmosphere ([1074de3](https://github.com/CoreySpohn/skyscapes/commit/1074de346a51caf16bea345e37ab56ddb5069098))
* **skyscapes:** add ParametricAtmosphere stub ([3c0a26a](https://github.com/CoreySpohn/skyscapes/commit/3c0a26ab300641fb1842e82db52d8e8a27c3d44b))
* **skyscapes:** add pooch-backed datasets module ([aa51caf](https://github.com/CoreySpohn/skyscapes/commit/aa51caf0b0d0cee4b80fb108ab07e8841880220d))
* **skyscapes:** add scene.AbstractStar, SimpleStar, SpectrumStar ([a439d10](https://github.com/CoreySpohn/skyscapes/commit/a439d10ab7a6d2aa399db5d101cb15b30cc4cf85))
* **skyscapes:** add scene.Planet/System, disk subpackage, ExoVista loader ([13f9916](https://github.com/CoreySpohn/skyscapes/commit/13f99160a21f61bb7a06e97217dd52b4ade92cab))


### Bug Fixes

* **skyscapes:** index ExoVista contrast grid by phase angle β ([df7f0d8](https://github.com/CoreySpohn/skyscapes/commit/df7f0d8d426b4865d85f8cbf1ad2984a18fb63cd))

## 0.0.1 (2026-04-13)


### Miscellaneous Chores

* release 0.0.1 ([574cbe9](https://github.com/CoreySpohn/skyscapes/commit/574cbe93db6b8150df6691d544306b3bc3fe1030))
