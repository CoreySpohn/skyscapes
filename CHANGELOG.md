# Changelog

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
