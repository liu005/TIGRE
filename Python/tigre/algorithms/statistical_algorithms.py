from __future__ import division

import copy
import numpy as np
from tigre.algorithms.iterative_recon_alg import IterativeReconAlg
from tigre.algorithms.iterative_recon_alg import decorator
from tigre.utilities.Atb import Atb
from tigre.utilities.Ax import Ax


class MLEM(IterativeReconAlg):  # noqa: D101
    __doc__ = (
        " MLEM solves the CBCT problem using the maximum likelihood expectation maximization\n"
        " algorithm\n"
        " \n"
        "  MLEM(PROJ,GEO,ANGLES,NITER,INIT) solves the reconstruction problem\n"
        "  using the projection data PROJ taken over ALPHA angles, corresponding\n"
        "  to the geometry described in GEO, using NITER iterations. INIT specifies\n"
        "  starting image, default: None (flat image value=1)"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        # Don't precompute V and W.
        kwargs.update(dict(W=None, V=None))
        kwargs.update(dict(blocksize=angles.shape[0]))
        # Nonnegativity is not optional for MLEM: the update is multiplicative,
        # so one negative factor flips a voxel's sign and it never recovers.
        # The old hand-rolled clip enforced this unconditionally, and routing
        # through apply_constraints must not quietly make it a user option.
        kwargs.update(dict(noneg=True))
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)

        if self.init is None:
            self.res += 1.0

        self.W = Atb(np.ones(proj.shape, dtype=np.float32), geo, angles, backprojection_type="matched", gpuids=self.gpuids)
        self.W[self.W <= 0.0] = np.inf

    # override
    def run_main_iter(self):
        # apply_constraints(), not a hand-rolled nonnegativity clip. This class
        # overrides run_main_iter and never routes through art_data_minimizing,
        # which is where the base class applies the feasible-set projection -
        # so before this, the mu_max ceiling and the known-air support mask
        # were applied to MLEM's WARM START and then never again, for the whole
        # run. Measured: with mu_max=0.5 on a 64^3 phantom, MLEM returned 3.46
        # while SIRT/OS_SART/FAST_OS_SART/ASD_POCS all returned exactly 0.5.
        # That is why an attenuation ceiling never tamed MLEM's hot voxels.
        self.apply_constraints()
        for i in range(self.niter):
            if self.Quameasopts is not None:
                res_prev = copy.deepcopy(self.res)
            self._estimate_time_until_completion(i)

            den = Ax(self.res, self.geo, self.angles, "interpolated", gpuids=self.gpuids)
            den[den == 0.0] = np.inf
            auxmlem = self.proj / den

            # update
            img = Atb(auxmlem, self.geo, self.angles, backprojection_type="matched", gpuids=self.gpuids) / self.W  

            self.res = self.res * img
            self.apply_constraints()
            if self.Quameasopts is not None:
                self.error_measurement(res_prev, i)


mlem = decorator(MLEM, name="mlem")
