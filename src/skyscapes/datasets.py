"""Data management utilities for skyscapes.

Pooch-managed fetchers for test/example FITS fixtures. Mirrors the
``scenes.zip`` entry from ``coronagraphoto.datasets`` (same file, hash,
and URL) but caches under its own ``skyscapes`` directory.

The PHOENIX and ATLAS9 stellar model atmospheres are exceptions to the
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


# Castelli and Kurucz (2004), solar metallicity, STScI cdbs ck04models.
# SHA-256 computed from all 76 served FITS files on 2026-11-13.
_CK04_HASHES = {
    3500: "sha256:6e87996333f01b9635d413adc6bb5b894deb875c79d823881ef03604e4f3f561",
    3750: "sha256:465386dc35a1107595216b91c148bb0fc4500bb247216a29260f126a9078b735",
    4000: "sha256:9c070d9b356aafd71384a6ec726ffe07de26c8cd5bc5bd6b22f4ac9418ddbfa7",
    4250: "sha256:7288ee6f093ab496ccb5e1420de8d22245fd5ae40ed7a34131b639ed9ab592c2",
    4500: "sha256:f8d3ed91b1a676a019ed46b2154c22dd30900fe26e550f0fb389629ddbe413a0",
    4750: "sha256:4774d088b744f61684b4f678370340959618b0037ba9b5af61605f63e82623fc",
    5000: "sha256:d217b79d347b0b1b96b8e4f8e2524340db2daf626c20750ba9c127c11172e1e3",
    5250: "sha256:9ef26e45b90cdccd5e14e03253661d711a5f3142dd647bbd75e4165e7db8662f",
    5500: "sha256:f291fd041added7ab06537a34f6d8c618db923d77f4d86b680040b7aeae9d4ac",
    5750: "sha256:d9de1de06b7cb9d5737f8217f3fa0bb615154c609bd56a602aafa71fad6942da",
    6000: "sha256:169117c6be4d758c982fe729526f65f8351ef49945be5a9e58a0a7aba6f00852",
    6250: "sha256:51a7422935fa74c8cd2330a02c1633c9ffff5afc80543be6d79e4720d4e77de4",
    6500: "sha256:504003ffc4d7242924277c496565cc81808353e4adbc75dd70bb09587fce0099",
    6750: "sha256:40482a32699b621d46d3363b55f96888b2c904659279fde9c551eccd4210302f",
    7000: "sha256:a17c7b197f571e5a23db6d147a291f001a41b07c11a586b96e75005e6667629e",
    7250: "sha256:9c478414db3a136a02abfdb731e52cd09ed88967498f8ca16f6a07def1429dd2",
    7500: "sha256:0966d8130ab6c45b23ab28cb0e25f3dcb80ba8481a0523ec03724cb2872b8a07",
    7750: "sha256:77db5b29cf65767255d4b5e502b6e322d7a3ee6ef92aa1a4ae07a845154cd687",
    8000: "sha256:f96da709fbc367921dbaa6b702880cff24ceec8d31f65e186047cbd489ddd7a0",
    8250: "sha256:484850229fcdfe400067622035a0de6dd07183b9604e8a16a2927205b333d231",
    8500: "sha256:5ab66f780fb72257080c3f22e5ba507d004d8d658a5edbaca1727e351d373086",
    8750: "sha256:516e87362384d4231d6d3dfe4489b2aa0cf6ad298464cfff0a4040027ab72734",
    9000: "sha256:4abc7d3f0ace3a112ca456b2e2590aa5f72521fbe7988ad166945e5d89535cdd",
    9250: "sha256:f5d32be427bc4ee1a2d6d40dfd0dafaac81817ca02a811898342cb582afbec51",
    9500: "sha256:767f961858d2914dfb6b8d29a6d34cdd9072434c26880ff0358fab673fbae28b",
    9750: "sha256:df73614cd64919340df4c31a1fa38b28b445e94aba5da462aa0107ddedd00bd8",
    10000: "sha256:cfc559b8d6477d12de8eeb67f40ebad6f004a92e842c9e2f3347c73e4ba39274",
    10250: "sha256:49216e4fef46fe5e021ea0adfd0c2e07f33fd7bdf2961433ccd2b5b9fd1ee360",
    10500: "sha256:559079b91c2ed75da441eb6e23a174cf7193a3fc436a5a2a450ca90b50cf3641",
    10750: "sha256:051d5089fae652bb82b3d55cefb29dad0661d7e4a7c6ca043726f0475599921d",
    11000: "sha256:10e42e149de04fab123061c8aab26bbd9bfa04aed031cd8b0fb298a15d7b6a4e",
    11250: "sha256:e5023c8212340d65455c7467e0f7d23dfc4930b05b64e045c4f898b502211253",
    11500: "sha256:eca05c9542a0e2d6861c50a15b2d08b226ee47d3aef84c090f214efcd20e89dc",
    11750: "sha256:7330026b6a75915098dead4077266cd20d0e8b3c4b2abef48a83766c85d7407f",
    12000: "sha256:394bc4f5be96c232df0b5d711800dd58bb5b0d4675928764864a9b7b8861b4f2",
    12250: "sha256:8ec98f312ba900ff88c97e65562ecc0f7424d3ae9d0ed3d911fe56ad193f9084",
    12500: "sha256:5da5c270aa0af8483317174f019c0784729a168cf474599ddebde036caf50a11",
    12750: "sha256:96bfa0d2cffd0e57206207aa0dfe7aa5d3235c63db6ac17432ec206d7a83d070",
    13000: "sha256:ebbf70268670478dab763a950bf14417f8bbf30ec919202eececbb7610241acf",
    14000: "sha256:2c7d9fba357a5d75586e301982aaf6bcab0ca87b764db9f4fa546efcffac9f16",
    15000: "sha256:ea89f7543809cb6bf00f2b7acd42d1716011ebff404e8b2bb9568d58c57bbe6c",
    16000: "sha256:486d987717aa08ada3505156a48a9f055b2244e0272c83ffc96d0965bcd14ebc",
    17000: "sha256:e763af21554f044dc267f74e41a8ba68b877275521ae26e9a393cb7c54572509",
    18000: "sha256:4fb9a644814cf7e7311b03f7f49c302c5bae0f4a06c47c5e9222c354aa90ac35",
    19000: "sha256:987e3364cb5cfea7710d5cca9cee4f8b44890cbb1e12d5de5b847162965e47e1",
    20000: "sha256:94df493d3d3e5a08a4cf8a078cae3a2c184c8641756e40bf1df43a98fc0477ce",
    21000: "sha256:8440d1b7e57a0488207e966e78bc67aa79b04f03196cb75db00e1ba4bd2ab745",
    22000: "sha256:8f40213b19ba9d251920b5f8e5fd363da46e1669304bc8abd6bf0cffe5d005d0",
    23000: "sha256:f51d9c9b134322da773b35e1a0aec4eff333255ba8e2db122068954b15034ec8",
    24000: "sha256:d10ed8be5d042d594fe207035cf70173eef84f3a9931222c447bab81d60001b9",
    25000: "sha256:5b0a4340c7784a543c262b3dc026b288e8ccffecab12dc410740f2d04ff1c89f",
    26000: "sha256:168305b2e273feb2d9a78f43b6bc33b4fb27ee61767012db9f0b74b3b4e1daa0",
    27000: "sha256:717fe9d714214e42d1d62badfb8d386fdc690c0ecdcb17c133dcd257ea01809f",
    28000: "sha256:ad5eac453fdd3cc60901fec865aa07d092696e6dee1ee5a2f2fee680a567f9e7",
    29000: "sha256:4ec55e8912fa9f3ae18190bb36027ef087a4e6f822b56fa0b6c2d292447392f7",
    30000: "sha256:ed4342e808e91b2552a2752d5bda8410ce465c14733e05b5b4ad6d9083850da6",
    31000: "sha256:16f700f110f4ce2db8a426e1ee2425f73e2d4c110620ed65b7bbf41862ce04a4",
    32000: "sha256:ce0dba35565af890d821296551494040c0e3d9a16643143c515c727037d53e8c",
    33000: "sha256:642222e40991bfc06402a6946927c7c38a420572d93f3e379008d7dc1db662cc",
    34000: "sha256:56cf5df03a94dce4d261c2f8dc85198112b51b133a796627466b79847f53143d",
    35000: "sha256:932f203f0b3d68e85af5f859210790d696603bcb5492a986ca97408b4da5fb43",
    36000: "sha256:8f4acc056dcbd67004f13f8dae17df0fb9fbb614ef2951af13904e6dd435a399",
    37000: "sha256:505fc5a387c66abbc25fa5f6186eab6465cf9e2a06143596c8ce08bfe8291617",
    38000: "sha256:7cc9366b3ab3f9201ea6b043023585bccf72358d5d30bc76e8e8922dad1f1da0",
    39000: "sha256:c291213a0a41a6a1f3aca7fae9a18e8ef44ddd9af267a879e9f5d821fa2cfea3",
    40000: "sha256:fef387c0d694570190c97af692ce7b547caa2174c94eef9ae958a66851e8dfde",
    41000: "sha256:d8a33e4437eadd95a4d47eb80254063404bfa5e8090e2ae61462de94fb85aace",
    42000: "sha256:07bf78d8cde2df00967be9569cde5e211df7d342cec8b4125487de3d4b2a364e",
    43000: "sha256:da2353eddd59fb819b7e374f53175774e53f29a6d41297c375c3f18e5dfa021e",
    44000: "sha256:40177255ed85117fa899a30f8e4767c7d60bde8f91f1227759ef55eea07828fa",
    45000: "sha256:a0534dc9f7f8393d9d5e4507c9e423d27e2d73e7eaff78a14a81aa2b41cb0529",
    46000: "sha256:755d4c3049bf5be3ef0ef188f20b834b7c275d5038ce8d897ca8287e0e1cd2b7",
    47000: "sha256:fcc6f71813e2c325dfed03825af33c3f8b2033f3b2cd648f3ffc20fb2f8616db",
    48000: "sha256:c924da5074b1308a59ead17405a914f529de1cdf8cd5f0eff25d3de2cec0fc30",
    49000: "sha256:9ba2eb9f8ccdd0b5d375ab263e60ec997b26c3eb4634dc6f99ae5be6b8e869cd",
    50000: "sha256:ca4eed9e57aeafd4ac91d6cfd353b40f3986a96cbff6b1876e535a6b8d45d962",
}
CK04_TEFF_K = tuple(_CK04_HASHES)
_CK04_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/"
    "ckp00/ckp00_{teff_k}.fits"
)


def _ck04_name(teff_k: int) -> str:
    return f"ckp00-{teff_k}-v1.fits"


REGISTRY = {
    **{_ck04_name(t): h for t, h in _CK04_HASHES.items()},
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
    **{_phoenix_name(t): h for t, h in _PHOENIX_HASHES.items()},
}

_URLS = {_phoenix_name(t): _PHOENIX_URL.format(teff_k=t) for t in _PHOENIX_HASHES}

_URLS.update({_ck04_name(t): _CK04_URL.format(teff_k=t) for t in _CK04_HASHES})

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


def list_ck04() -> list[int]:
    """List solar-metallicity ATLAS9 Teff nodes [K], from 3500 to 50000."""
    return list(CK04_TEFF_K)


def fetch_ck04(teff_k: int) -> str:
    """Fetch a hash-verified Castelli and Kurucz (2004) ATLAS9 FITS node.

    Each file contains all gravity columns, including zero-filled columns
    for gravities unavailable at this temperature. See ``list_ck04`` for
    registered temperatures in kelvin. Downloads are cached by pooch.
    """
    if teff_k not in _CK04_HASHES:
        raise ValueError(f"teff_k = {teff_k} is not a registered CK04 node.")
    return str(PIKACHU.fetch(_ck04_name(teff_k)))
