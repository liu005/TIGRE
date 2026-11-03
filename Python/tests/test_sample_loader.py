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
