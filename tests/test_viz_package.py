"""skyscapes.viz packaging: optional extra, lazy exports, no import side effects."""

from __future__ import annotations

import subprocess
import sys

import pytest

BLOCK_EYEPIECE = "import sys; sys.modules['eyepiece'] = None; "


def _run(code):
    """Run ``code`` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_library_imports_without_eyepiece():
    """The base install imports clean, viz package included, with no eyepiece."""
    result = _run(BLOCK_EYEPIECE + "import skyscapes; import skyscapes.viz")
    assert result.returncode == 0, result.stderr


def test_touching_a_view_without_eyepiece_names_the_extra():
    """A view looked up without eyepiece raises with the install hint."""
    code = (
        BLOCK_EYEPIECE + "import skyscapes.viz as viz\n"
        "try:\n"
        "    viz.plot_system\n"
        "except ImportError as err:\n"
        "    assert 'skyscapes[viz]' in str(err), err\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_viz_does_not_import_eyepiece():
    """The package itself is lazy: eyepiece loads only when a view is used."""
    code = (
        "import sys, skyscapes.viz\n"
        "assert 'eyepiece' not in sys.modules, 'eyepiece imported eagerly'\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_resolving_every_view_leaves_rcparams_alone():
    """Importing every view module changes no matplotlib global state."""
    code = (
        "import matplotlib as mpl\n"
        "before = dict(mpl.rcParams)\n"
        "import skyscapes.viz as viz\n"
        "for name in viz.__all__:\n"
        "    getattr(viz, name)\n"
        "after = dict(mpl.rcParams)\n"
        "changed = [k for k in before if before[k] != after[k]]\n"
        "assert not changed, changed\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exports_are_listed_and_unknown_names_raise():
    """``__all__`` and ``dir`` list the views; an unknown name raises."""
    import skyscapes.viz as viz

    expected = {
        "plot_disk_geometry",
        "plot_disk_image",
        "plot_local_zodi_geometry",
        "plot_system",
    }
    assert set(viz.__all__) == expected
    assert expected <= set(dir(viz))
    with pytest.raises(AttributeError):
        viz.plot_nothing  # noqa: B018
