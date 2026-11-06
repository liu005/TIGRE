"""Check the aligned demo acquisitions and the real GUI archive workflow."""
import ast
from pathlib import Path
import runpy
import numpy as np
import pytest
import tigre
from test_detector_geometry_examples import _demo_geometry

DEMOS = Path(__file__).resolve().parents[1] / "demos"


@pytest.mark.parametrize("name,nvoxel,ndetector,nviews", [
    ("d03_generateData.py", (256, 256, 256), (256, 256), 100),
    ("d06_Algorithms01.py", (128, 128, 128), (256, 256), 200),
    ("d10_QualityMeasurements.py", (128, 128, 128), (256, 256), 100),
    ("d13_HelicalGeometry.py", (512, 128, 128), (512, 512), 300),
    ("d15_3Dparallel.py", (128, 128, 128), (128, 128), 360),
])
def test_acquisition_matches_matlab_example(name, nvoxel, ndetector, nviews):
    # Run only geometry/angle assignments, before phantom loading or GPU calls.
    demo = _demo_geometry(name, "head")
    np.testing.assert_array_equal(demo["geo"].nVoxel, nvoxel)
    np.testing.assert_array_equal(demo["geo"].nDetector, ndetector)
    assert len(demo["angles"]) == nviews
    if name.startswith("d13"):
        np.testing.assert_allclose(demo["geo"].sVoxel, [1024, 256, 256])


def test_quality_demo_reconstructs_the_noisy_data_for_thirty_iterations():
    tree = ast.parse((DEMOS / "d10_QualityMeasurements.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr in ("sirt", "sart", "os_sart")]
    assert len(calls) == 3
    for call in calls:
        assert call.args[0].id == "noise_projections"
        assert ast.literal_eval(call.args[3]) == 30


def test_modular_demo_has_matlab_panel_tilts_cor_and_masked_gaps():
    demo = runpy.run_path(str(DEMOS / "d26_ModularDetector.py"))
    settings, surface = demo["build_surface"](5)
    assert surface.valid_mask.shape == (5, 80, 100)
    assert settings["cor_mm"] == 2
    assert settings["panel_x_angles_deg"] == (-3, 0, 3)
    assert settings["panel_y_angles_deg"] == (2, -2, 1)
    assert not surface.valid_mask[:, :, 32:34].any()
    assert not surface.valid_mask[:, :, 66:68].any()
