"""Importing skyscapes stays light: no ExoJAX or plotting stack until asked for."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("exojax", "matplotlib", "radis", "vaex")


def _run(code):
    """Run ``code`` in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_import_leaves_heavy_stack_out():
    """Package, scene and population imports load neither ExoJAX nor matplotlib."""
    code = (
        "import sys, skyscapes, skyscapes.scene, skyscapes.population\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exojax_names_load_lazily_through_the_package():
    """The public ExoJAX names resolve on first access and then pull exojax in."""
    code = (
        "import sys\n"
        "from skyscapes.physical_model import ExoJaxEmissionModel\n"
        "from skyscapes.physical_model import ExoJaxPhysicalModel\n"
        "import skyscapes.physical_model as pm\n"
        "assert 'exojax' in sys.modules\n"
        "assert ExoJaxPhysicalModel.__name__ == 'ExoJaxPhysicalModel'\n"
        "assert pm.ExoJaxEmissionModel is ExoJaxEmissionModel\n"
        "try:\n"
        "    pm.NoSuchModel\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('no AttributeError')\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr
