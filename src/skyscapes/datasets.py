"""Data management utilities for skyscapes.

Pooch-managed fetchers for test/example FITS fixtures. Mirrors the
``scenes.zip`` entry from ``coronagraphoto.datasets`` (same file, hash,
and URL) but caches under its own ``skyscapes`` directory.

The PHOENIX stellar model atmospheres are the one exception to the
"hosted in the owning repository" rule: the grid is an upstream STScI
reference-atlas product (about 11.5 MB per Teff node, 76 nodes), so every node
is registered with its SHA-256 hash and fetched lazily from the archive when
first requested. Registry names carry a ``-v1`` suffix; the bytes at the
upstream URL are not versioned by STScI, so a hash mismatch means upstream
changed and the suffix is bumped with the new hashes.
"""

from __future__ import annotations

import pooch
from pooch import Unzip

# Solar-metallicity (phoenixm00) nodes from the STScI cdbs grid, 2000 to 12000 K.
# Hashes were computed from the files as served on 2026-11-07.
_PHOENIX_HASHES = {
    2000: "sha256:15a1101e52bd39989ca5e69fb0d0d3da0604cca545779289077fa6dd153aa44e",
    2100: "sha256:31eb72db2b777f8417d9d5a4167f4149330321c0fc40149b21b180c63dd90988",
    2200: "sha256:1aa63da091ac1bc21b4815247f2fd3683fa2c6c793111b3ed4898cb13c0b8d63",
    2300: "sha256:4d93fa7b3830b8c3f64f1696d66ced97f9bab9a519decf285ee9bfb9a2bda81b",
    2400: "sha256:29cfc81c59427cae40fe75416688d771a67d85cccab02606519ddaf3116fc82a",
    2500: "sha256:b01aa88bc8f72016e72ff7943e05f64dceea84ab14a47b19c8e8ca89ed25ce05",
    2600: "sha256:15841606a5191e01375dcef8637f1d6f4c1870bcbdbfb6f50b536276be65ea0d",
    2700: "sha256:d8142266dbd62b0bdf580ffdf0ffe0e6cad02a19895623e62e7383b0afe6057c",
    2800: "sha256:a0e9082bd0de33b1c1f5f2ad94874abb8b49dd6f0abb2a6e33c67bce96e048b6",
    2900: "sha256:8da2f63d7ae1cbe7325ea5ccc064a874aaf4bf1e7a1a7d1819908d33305a9b28",
    3000: "sha256:fd9ea3d54ce7ef5fa6e641af6b111254cd69abe8b508e101ab0d46d40140c76a",
    3100: "sha256:0bafb5795ca907bbc18116f6035b190f9253c3c32812d96e6608a3b040bbeb4e",
    3200: "sha256:34d6224b9ca3700d2c21f978cd87969476f33e33c436ea06eada493369c2001c",
    3300: "sha256:c247fe8035efb8b1c254d8e2966f94d50d0e295ba6301076af9b4bbd7122b9d0",
    3400: "sha256:caa2701d1edab228b5893c803889bb5a486d1197ccc8a00321f8a10ee7ee06d3",
    3500: "sha256:eb3355d78a00263c567b273727ed90e2a198e4f022bb7f733763177fae862600",
    3600: "sha256:8656e854decd244977737c11b5f2ff7e558c535fafe2dc9d61c1075b8e8ff538",
    3700: "sha256:69270b1435e3ed04ad68223a7d1c2b9e58bfdb98211e1c9a8d8e1ec90647a8c7",
    3800: "sha256:4cb17984ef462dfecaad793d0771a42752cde1198cb26355042b9c27922057a6",
    3900: "sha256:bd8819477a146ad3eef5ec4431123d87233685a0f9283dc0b1e4a726d9f5c5ec",
    4000: "sha256:ce0d76ce405b9d28dc4089d997867f8433fd6f2d84007c0367a8b72f7c2a8f05",
    4100: "sha256:3cc50e608f7d889f32e73e8936d5a04559cb430ffdcc4dbfc88112a3243707d1",
    4200: "sha256:c871b11a4fad91b429470455e4b024febf10dc770c35e26d52be821ac8fa07db",
    4300: "sha256:b6b225b679dee29a6d860e0264599ffb58987073ec9bf967d26ef40d5ce23463",
    4400: "sha256:e8f6e3be0eb37bb7d748a882381139990758eef01d823c28d0ff800ab324f0e3",
    4500: "sha256:729fda742382b1d0ff58da63c55daab7f02423e41f8718df7973fb951fc05b9c",
    4600: "sha256:4cf7e22e1685958c28809869e0cee9039cd814097a0b49ddacbec7f80a75514f",
    4700: "sha256:a4a388489d2b179f60485bce79cab6eaef2b08a10b12430133a37ca8074effb6",
    4800: "sha256:bbe30e3617449f0da4ab1f43aa83b2697ef064707ae2dac5fcf325d1a07cb6b8",
    4900: "sha256:c7480e419ec9bc04f5334a146de09cc95249f44648333a0add7d7756f4031c48",
    5000: "sha256:271d278b0d7eb10fb5733298ede4f27ddc0a2c52f311167d67319b11d3f1185d",
    5100: "sha256:55230a36679a3621b776c3c8c04660b0b7155b3274f017baead84f17da0f8e71",
    5200: "sha256:5bbfb5018ca3068146ec7a45c8952f9819a70eb666351c73936ec9410f615733",
    5300: "sha256:451018eb13272e20edf0a7cf0db3ea15ba4632a63b3e8a095cf8ca5dc36bc470",
    5400: "sha256:b69579bbaa30b1b6d936e7cd7e70ba785446051b249a02730f9867226ef1ba81",
    5500: "sha256:386ae15e6c1e1fe22389d4fc612c0718853f4cb0265def4e42f6015f047d4866",
    5600: "sha256:688a5acf69c9ec65a1b5398084d6c62e43ad63d2799588e5a005f715123686ca",
    5700: "sha256:373a945f3c16f36473fdca542d5a97c090a16871ad598f7b069fd15bb4bbe027",
    5800: "sha256:4da21db675bca8929b8a87194a86fc004f0517e777b5d75617d534b1b5d79008",
    5900: "sha256:c104a14034f7fae56dcab831461b5b4763dc7f9bd2948900a4c73032909aa991",
    6000: "sha256:3205d53847889b3d3ee687adef044a624bcd79989a2e471b388834cbbd7334d2",
    6100: "sha256:d6b51c8b7e746388144912d1542d29d58823a3b9792edcc0b83506548afd8b6b",
    6200: "sha256:49d2f7b84acee4ead795f093e55ceb5292947af20958ef265fde334dd881f9ce",
    6300: "sha256:05570ac5094f8137d0ca39029a2fcf3420e777f75637d527805e69e017cf2b2d",
    6400: "sha256:6dc80c1b428a96ea4b98bae65ae8673400001f2adcdef753fa73449ecde3a6e5",
    6500: "sha256:9095fd33f9501e0af2e26c01271c38bd5bf8431a12aecb53f71b3aada6d600a0",
    6600: "sha256:b05f8a9fa70fcd7b81c7128e7400cf25cdcfd46219322a270e7769b04bbf5c8e",
    6700: "sha256:6f07d6f5b99969c9e63f75f830e94ced21d1986d09a87ca43281772e7f668406",
    6800: "sha256:d3d07af3f8c18d4f13122e15ea5ee4b3dd336ea40df58974af000b6ed2916c28",
    6900: "sha256:3a7965aa5c130873c50506571b1ae06de63f8da529533710790d0bc77c7d6c82",
    7000: "sha256:057f2960232e6fcba3920343d29d19331cd035389700ce08b094f3112172505c",
    7200: "sha256:44227e436379732b78f48f4d3d9abdbf3edbba64f2b034e20d4ece668013ca94",
    7400: "sha256:c677b1e75df5083490735b122cfe6250d72295619e52ce4985ae777faa7b3556",
    7600: "sha256:cfb0765fd99d34f8c6bc6ac7bf9ff2b5e4c031ee29b6acbaec165503134d9783",
    7800: "sha256:90b2fece94be280606731180a7242769a11130ec38c55d13e3977202b3baadac",
    8000: "sha256:edafc51f0f5305a9ba58629ffb7a1cdf4cbcf8e8c2dc01cc9606c5c0da6bcaef",
    8200: "sha256:9154c4bf154cd1f8dd3d599e93f4fd27b6d3fffe555bb1c8359f9391c3533af9",
    8400: "sha256:af056acded0101857c7fb967b47fbed11ecac6c890fafb7affec14c423a0dff8",
    8600: "sha256:eb31f7476423de214e26e2c9973f36fd16dacf80f683e002cbe8b0ad8eaddc17",
    8800: "sha256:9334830d4280bd124fa3fa53466c653ca299bb47ae846a06941ab94c9518957b",
    9000: "sha256:b0fca451b4517834bfa2e5f71d9f80e75ff209d9c765e02f8d3590428503ddf0",
    9200: "sha256:267941dea842bdc8f994588316cd772208c972c75d41d47973b9563c33936652",
    9400: "sha256:f901360b86c0b665b2451daebf45b2fcf2484f9f66676792811cdea5abc01404",
    9600: "sha256:13faef61d91969c4b2d10d3c9b15642e732db92cf63a119096e0be23eff29209",
    9800: "sha256:7c8aec71d8c5f4f7169638ddf4b0679ae4b23163525ee0e48bf043c547d7257d",
    10000: "sha256:51ba5cc5e64849e18060b268fd31ee4f23b0bc78120d97c32b9254fcd010b20d",
    10200: "sha256:67915b58dbc2f8a3caaada9799ccd4a341764789e38d4ce527e56fa926a7cf02",
    10400: "sha256:b7f3a0ee1788d64f9c1a38f5c0fcbc13fe78e31f33f62726a7fc55b177e5ca74",
    10600: "sha256:c5066f113380e2cb0ff2e3b6de06dd80634b547798e3dbd13078d6d4ac02dedf",
    10800: "sha256:3cb1672e914654621e2a4f5ebeb7d5e1c2844df11d23e9fcbee79b49392938e6",
    11000: "sha256:17e3d24312ffef9c417271934881999cdb7a9a510d75e47361839c5b712aaf79",
    11200: "sha256:080ca26dcae56508ba2edaa9e9e6c107e2831cf0b9328b2dbe62d2f441b7b6f6",
    11400: "sha256:c425231ecd3d161a7fa7057f6b246a4dd87f901809a04d62ef13c90509b2a14a",
    11600: "sha256:7905d0406a545bd3339ffe9d8799ca0b59ece5a999b1c631457ab878949b6552",
    11800: "sha256:2016eaadfae290de8f64d83bc0e739b8998dc49cacfc229aa2373c9d99b34fce",
    12000: "sha256:73de62f6bfe835993ddd079d3c30cc6d72076a099d4aae36785625da28009b65",
}
PHOENIX_TEFF_K = tuple(_PHOENIX_HASHES)
_PHOENIX_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/"
    "phoenixm00/phoenixm00_{teff_k}.fits"
)


def _phoenix_name(teff_k: int) -> str:
    return f"phoenixm00-{teff_k}-v1.fits"


REGISTRY = {
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
    **{_phoenix_name(t): h for t, h in _PHOENIX_HASHES.items()},
}

_URLS = {_phoenix_name(t): _PHOENIX_URL.format(teff_k=t) for t in _PHOENIX_HASHES}

PIKACHU = pooch.create(
    path=pooch.os_cache("skyscapes"),
    base_url="https://github.com/CoreySpohn/coronalyze/raw/main/data/",
    registry=REGISTRY,
    urls=_URLS,
)


def fetch_scene() -> str:
    """Fetch and unpack the canonical ExoVista demo scene.

    Returns:
        Absolute path to ``solar_system_mod.fits``.
    """
    PIKACHU.fetch("scenes.zip", processor=Unzip())
    return str(
        PIKACHU.abspath / "scenes.zip.unzip" / "scenes" / "solar_system_mod.fits"
    )


def list_phoenix() -> list[int]:
    """List the registered PHOENIX Teff nodes in kelvin (solar metallicity)."""
    return list(PHOENIX_TEFF_K)


def fetch_phoenix(teff_k: int) -> str:
    """Fetch one PHOENIX model-atmosphere node from the STScI archive.

    The file holds every tabulated ``logg`` for that Teff. It is downloaded on
    first use, verified against the registry hash, and cached.

    Args:
        teff_k: Node effective temperature in kelvin; one of
            ``list_phoenix()``.

    Returns:
        Absolute path to ``phoenixm00_<teff_k>.fits`` in the cache.

    Raises:
        ValueError: If ``teff_k`` is not a registered node.
    """
    if teff_k not in _PHOENIX_HASHES:
        raise ValueError(f"teff_k = {teff_k} is not a registered PHOENIX node.")
    return str(PIKACHU.fetch(_phoenix_name(teff_k)))
