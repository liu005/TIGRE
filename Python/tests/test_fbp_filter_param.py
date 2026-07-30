import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_TIGRE_PKG = "/Users/csl/Desktop/TIGRE/Python"
if _TIGRE_PKG not in sys.path:
    sys.path.insert(0, _TIGRE_PKG)

for _mod in [
    "_Ax", "_Atb", "_tv_proximal", "_AwminTV", "_minTV", "_minPICCS",
    "_gpuUtils", "_RandomNumberGenerator",
]:
    sys.modules[_mod] = MagicMock()

for _mod in ["tigre.utilities.Ax", "tigre.utilities.gpu"]:
    sys.modules[_mod] = MagicMock()

from tigre.utilities.geometry import Geometry


class TestFBPPFilterPropagation(unittest.TestCase):

    def _make_geo(self):
        geo = Geometry()
        geo.mode = "parallel"
        geo.DSD = np.array([1000.0], dtype=np.float32)
        geo.DSO = np.array([500.0], dtype=np.float32)
        geo.nVoxel = np.array([64, 64, 64], dtype=np.int32)
        geo.sVoxel = np.array([64, 64, 64], dtype=np.float32)
        geo.dVoxel = np.ones(3, dtype=np.float32)
        geo.nDetector = np.array([128, 128], dtype=np.int32)
        geo.sDetector = np.array([128, 128], dtype=np.float32)
        geo.dDetector = np.ones(2, dtype=np.float32)
        return geo

    @patch("tigre.algorithms.single_pass_algorithms.Atb", autospec=True)
    @patch("tigre.algorithms.single_pass_algorithms.filtering", autospec=True)
    def test_filter_kwarg_goes_to_geox_not_geo(self, mock_filtering, mock_Atb):
        from tigre.algorithms.single_pass_algorithms import fbp

        geo = self._make_geo()
        proj = np.zeros((4, 128, 128), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False, dtype=np.float32)

        mock_filtering.return_value = proj
        mock_Atb.return_value = np.zeros((64, 64, 64), dtype=np.float32)

        fbp(proj, geo, angles, filter="hann")

        called_geo = mock_filtering.call_args[0][1]
        self.assertIsNot(called_geo, geo)
        self.assertEqual(called_geo.filter, "hann")
        self.assertIsNone(geo.filter)

    @patch("tigre.algorithms.single_pass_algorithms.Atb", autospec=True)
    @patch("tigre.algorithms.single_pass_algorithms.filtering", autospec=True)
    def test_fbp_default_filter_is_none(self, mock_filtering, mock_Atb):
        from tigre.algorithms.single_pass_algorithms import fbp

        geo = self._make_geo()
        proj = np.zeros((4, 128, 128), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False, dtype=np.float32)

        mock_filtering.return_value = proj
        mock_Atb.return_value = np.zeros((64, 64, 64), dtype=np.float32)

        fbp(proj, geo, angles)

        called_geo = mock_filtering.call_args[0][1]
        self.assertIsNot(called_geo, geo)
        self.assertIsNone(called_geo.filter)


if __name__ == "__main__":
    unittest.main()
