"""Bundled phantom failures remain actionable and resampling preserves shape."""
import numpy as np
import pytest
from tigre.utilities import sample_loader


def test_missing_head_phantom_has_an_actionable_error(monkeypatch):
    monkeypatch.setattr(sample_loader.os.path, "isfile", lambda _path: False)
    with pytest.raises(FileNotFoundError, match="Common/data/head.mat"):
        sample_loader.load_head_phantom([8, 10, 12])


def test_head_phantom_supports_anisotropic_voxel_counts():
    head = sample_loader.load_head_phantom(np.array([8, 10, 12]))
    assert head.shape == (8, 10, 12)
    assert np.isfinite(head).all()
    assert np.ptp(head) > 0


def test_installed_phantom_in_environment_data_directory(tmp_path, monkeypatch):
    """Linux wheel data lives outside lib/pythonX.Y/site-packages."""
    prefix = tmp_path / "env"
    module = prefix / "lib/python3.8/site-packages/tigre/utilities/sample_loader.py"
    module.parent.mkdir(parents=True)
    data = prefix / "data"
    data.mkdir()
    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    sample_loader.scipy.io.savemat(data / "head.mat", {"img": image})
    monkeypatch.setattr(sample_loader, "__file__", str(module))
    monkeypatch.setattr(sample_loader.sys, "prefix", str(prefix))

    result = sample_loader.load_head_phantom([4, 3, 2])
    np.testing.assert_allclose(result, sample_loader.scipy.ndimage.zoom(
        image.transpose(2, 1, 0), (1, 1, 1), order=3, prefilter=False
    ))
