"""The declared floors on sibling libraries keep old and new releases apart.

skyscapes 2.1.0 reads orbix's (east, north, away) states, so it must not
install beside an orbix that returns (north, east, away). The pins live in
``pyproject.toml``; this test reads them without installing anything.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# First release of each sibling that uses the (east, north, away) sky chart.
SKY_FLOORS = {
    "orbix": Version("3.0.0"),
    "photomancy": Version("0.4.0"),
    "coronagraphoto": Version("4.2.0"),
    "coronalyze": Version("1.2.0"),
}


def _requirements():
    """Yield (group, Requirement) for every declared dependency."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    for text in project["dependencies"]:
        yield "dependencies", Requirement(text)
    for extra, texts in project.get("optional-dependencies", {}).items():
        for text in texts:
            yield f"extra:{extra}", Requirement(text)


def _floor(requirement):
    lowers = [
        Version(s.version) for s in requirement.specifier if s.operator in (">=", "==")
    ]
    return max(lowers) if lowers else None


def test_orbix_floor_is_declared_for_install_and_test():
    """Both the runtime dependency and the test extra require orbix 3.0.0."""
    groups = {
        group: _floor(req) for group, req in _requirements() if req.name == "orbix"
    }
    assert groups["dependencies"] == SKY_FLOORS["orbix"]
    assert groups["extra:test"] == SKY_FLOORS["orbix"]


def test_every_sibling_pin_is_at_or_above_the_sky_chart_release():
    """No dependency, extra or docs group names an older sibling than the train."""
    for group, req in _requirements():
        if req.name in SKY_FLOORS:
            floor = _floor(req)
            assert floor is not None, f"{req.name} is unpinned in {group}"
            assert floor >= SKY_FLOORS[req.name], f"{req.name} floor in {group}"
