import importlib.util
import sys
from pathlib import Path
from types import ModuleType

VARIAN_DIR = Path(__file__).resolve().parents[1] / "tigre" / "utilities" / "io" / "varian"


def _load_varian_io(monkeypatch):
    for name in (
        "tigre",
        "tigre.utilities",
        "tigre.utilities.io",
        "tigre.utilities.io.varian",
    ):
        package = ModuleType(name)
        package.__path__ = [str(VARIAN_DIR)]
        monkeypatch.setitem(sys.modules, name, package)

    geometry = ModuleType("tigre.utilities.geometry")
    geometry.Geometry = object
    monkeypatch.setitem(sys.modules, geometry.__name__, geometry)

    xim = ModuleType("tigre.utilities.io.varian.xim")
    xim.XIM = object
    monkeypatch.setitem(sys.modules, xim.__name__, xim)

    utils_name = "tigre.utilities.io.varian.utils"
    utils_spec = importlib.util.spec_from_file_location(utils_name, VARIAN_DIR / "utils.py")
    utils = importlib.util.module_from_spec(utils_spec)
    monkeypatch.setitem(sys.modules, utils_name, utils)
    utils_spec.loader.exec_module(utils)

    varian_name = "tigre.utilities.io.varian.varian_io"
    varian_spec = importlib.util.spec_from_file_location(varian_name, VARIAN_DIR / "varian_io.py")
    varian_io = importlib.util.module_from_spec(varian_spec)
    monkeypatch.setitem(sys.modules, varian_name, varian_io)
    varian_spec.loader.exec_module(varian_io)
    return varian_io


def test_rot_direction_uses_the_sign_of_the_angle_delta(monkeypatch):
    scan_params = _load_varian_io(monkeypatch).ScanParams

    counterclockwise = object.__new__(scan_params)
    counterclockwise.start_angle = 0.0
    counterclockwise.stop_angle = 10.0

    clockwise = object.__new__(scan_params)
    clockwise.start_angle = 10.0
    clockwise.stop_angle = 0.0

    assert counterclockwise.rot_direction() == "CC"
    assert clockwise.rot_direction() == "CW"
