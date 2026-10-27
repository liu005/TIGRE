"""Verify the shipped YAML-to-CUDA reconstruction workflow and saved results."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('_AxSurface')


@pytest.mark.parametrize("phantom, tolerance", [("head", 0.05), ("ellipsoids", 0.01)])
def test_yaml_reconstruction_end_to_end(tmp_path, phantom, tolerance):
    path = Path(__file__).resolve().parents[1] / 'demos' / 'd_geometry_yaml_reconstruction.py'
    spec = importlib.util.spec_from_file_location('yaml_reconstruction_demo', path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    geometry = path.parent / 'geometry_examples' / 'three_panel.yaml'
    metrics = demo.run(geometry, tmp_path, iterations=100, phantom=phantom)
    assert metrics['phantom_relative_l2_error'] < tolerance
    assert metrics['projection_relative_l2_residual'] < 0.001
    saved_metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert saved_metrics['completed_iterations'] == 100
    with np.load(tmp_path / 'results.npz', allow_pickle=False) as result:
        assert result['projections'].shape == (48, 16, 100)
        assert result['phantom'].shape == (16, 32, 32)
        if phantom == "head":
            from tigre.utilities.sample_loader import load_head_phantom
            np.testing.assert_array_equal(result['phantom'], load_head_phantom(np.array([16, 32, 32])).astype(np.float32))
        assert len(result['module_slices']) == 3
        assert not result['valid_mask'].all()
        assert np.all(result['projections'][~result['valid_mask']] == 0)
        relative_error = np.linalg.norm(result['reconstruction']-result['phantom']) / np.linalg.norm(result['phantom'])
        np.testing.assert_allclose(relative_error, metrics['phantom_relative_l2_error'])
        assert result['residuals'][-1] < result['residuals'][0] * 0.001
    assert (tmp_path / 'comparison.png').stat().st_size > 1000
