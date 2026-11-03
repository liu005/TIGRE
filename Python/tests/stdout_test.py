import io
from contextlib import redirect_stdout
import os
import sys
import unittest

import numpy as np
import tigre
import tigre.algorithms as algs
from tigre.utilities import sample_loader

dirname = os.path.dirname(__file__)


class TestStdout(unittest.TestCase):
    pass


def make_stdout_test(algorithm, proj, geo, angles, niter):
    def test(self):
        capturedOutput = io.StringIO()
        with redirect_stdout(capturedOutput):
            getattr(algs, algorithm)(proj, geo, angles, niter=niter, verbose=False)
        self.assertEqual(capturedOutput.getvalue(), "")

    return test


if __name__ == "__main__":
    geo = tigre.geometry(mode="cone", default=True, high_resolution=False)
    print(geo)

    true_img = sample_loader.load_head_phantom(geo.nVoxel)
    angles = np.linspace(0, 2 * np.pi, 100)
    niter = 5
    proj = tigre.Ax(true_img, geo, angles)
    for alg in algs.__all__:
        if alg != "fbp":
            test_name = "test_print_%s" % (alg)
            test = make_stdout_test(alg, proj, geo, angles, niter)
            setattr(TestStdout, test_name, test)

    unittest.main()
