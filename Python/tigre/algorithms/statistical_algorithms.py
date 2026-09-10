from __future__ import division

import copy
import warnings

import numpy as np
from tigre.algorithms.iterative_recon_alg import IterativeReconAlg
from tigre.algorithms.iterative_recon_alg import decorator
from tigre.utilities.Atb import Atb
from tigre.utilities.Ax import Ax

# Forward projections below this (attenuation-length units) carry no object
# information: the ray is treated like an empty one (ratio 0). A den of 1e-6 is
# a millionth of one attenuation length - far below anything a real ray through
# a real object produces - while a denormal float32 den (1e-40) turned into a
# ratio of 1e38, an inf after the backprojection sum, and NaN at the next
# multiply (see mlem_ratio).
MLEM_DEN_FLOOR = 1e-6
# Ceiling on the per-iteration multiplicative factor. On log-attenuation data a
# ray through empty space has den -> 0 and proj = +-noise; the positive half of
# those rays gives ratios of 1e4..1e8 and the voxels along them grow by that
# factor every iteration (a real 360-view run: finite volume with max 1.3e15
# after 10 iterations). A true update is within a few percent of 1 (warm start)
# and never above a few units; x10 per iteration leaves every genuine update
# untouched and bounds the pathological ones.
MLEM_RATIO_MAX = 10.0


def mlem_ratio(proj, den, eps=MLEM_DEN_FLOOR, ratio_max=MLEM_RATIO_MAX):
    """proj / den for the MLEM update, IN PLACE into den, made safe.

    The textbook guard `den[den == 0] = inf` catches only an exact zero. Under
    the multiplicative update the air voxels of a warm-started volume decay
    geometrically (a 64^3 phantom: smallest positive voxel 2e-20 after 10
    iterations, float32 denormal 1.4e-45 by 40), so a ray that sees only such
    voxels has a den of 1e-30..1e-45: proj/den is up to 1e38, the matched
    backprojection sums those to inf, and res * inf is inf for a voxel that is
    not zero and NaN for one that is (nonnegativity clip, support mask). One
    such ray poisons the whole volume within two iterations. Seen on a real
    360-view 1024^3 run with a support prior (2026-09-10): `overflow
    encountered in multiply`, then `invalid value`, and a [nan, nan] volume.

    Rays with den <= eps get ratio 0 (no information, like an empty ray), and
    negative data - log projections after scatter / seam corrections can be
    slightly negative in air - are clipped at 0, because a negative factor has
    no meaning in a multiplicative update. The ratio is capped at ratio_max
    (MLEM_RATIO_MAX): with the floor alone a near-empty ray still gives ratios
    of 1e4 and the voxels along it grow without bound (finite but 1e15 after
    ten iterations on real data). Elsewhere the ratio is unchanged.
    Returns den (now holding the ratio); no extra full-size buffer is made.
    """
    ok = den > eps
    np.divide(proj, den, out=den, where=ok)
    den[~ok] = 0.0
    np.clip(den, 0.0, ratio_max, out=den)
    return den


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
            auxmlem = mlem_ratio(self.proj, den)      # in place into den; see mlem_ratio

            # update
            img = Atb(auxmlem, self.geo, self.angles, backprojection_type="matched", gpuids=self.gpuids) / self.W
            del auxmlem, den

            self.res = self.res * img
            self.apply_constraints()
            if not np.isfinite(self.res).all():
                # Should be unreachable with mlem_ratio; if it ever happens the
                # rest of the iterations would only spread the damage.
                warnings.warn(f"MLEM: non-finite voxels after iteration {i + 1}; stopping "
                              f"(volume is NOT usable - check the data for inf/NaN and the "
                              f"warm start for negative or huge values)", RuntimeWarning)
                break
            if self.Quameasopts is not None:
                self.error_measurement(res_prev, i)


mlem = decorator(MLEM, name="mlem")
