from __future__ import division

import copy

import numpy as np
import tigre
from tigre.algorithms.iterative_recon_alg import IterativeReconAlg
from tigre.algorithms.iterative_recon_alg import decorator
from tigre.utilities.im_3d_denoise import im3ddenoise



class FISTA(IterativeReconAlg):
    """
    Solves the reconstruction problem
    using the projection data PROJ taken over ALPHA angles, correspond-
    ing to the geometry described in GEO, using NITER iterations.

    Parameters
    ----------
    :param proj: (np.ndarray, dtype=np.float32)
    Input data, shape = (geo.nDector, nangles)

    :param geo: (tigre.geometry)
    Geometry of detector and image (see examples/Demo code)

    :param angles: (np.ndarray , dtype=np.float32)
    angles of projection, shape = (nangles,3)

    :param niter: (int)
    number of iterations for reconstruction algorithm

    :param kwargs: (dict)
    optional parameters

    Keyword Arguments
    -----------------
    :keyword hyper: (np.float64)
        hyper parameter proportional to the largest eigenvalue of the
        matrix A in the equations Ax-b and ATb.
        Empirical tests show, for the headphantom object:

        nVoxel = np.array([64,64,64]),      hyper (approx=) 2.e8
        nVoxel = np.array([512,512,512]),   hyper (approx=) 2.e4

    :keyword init: (str)
        Describes different initialization techniques.
              "none"     : Initializes the image to zeros (default)
              "FDK"      : initializes image to FDK reconstruction
              
    :keyword verbose:  (Boolean)
        Feedback print statements for algorithm progress
        default=True

    :keyword OrderStrategy : (str)
        Chooses the subset ordering strategy. Options are:
                 "ordered"        : uses them in the input order, but
                                    divided
                 "random"         : orders them randomly

    :keyword tviter: (int)
        Number of iterations of im3ddenoise for every iteration.
        Default: 20

    :keyword tvlambda: (float)
        Multiplier for lambdaForTV which is proportional to L (hyper)
        Default: 0.1      
        
    :keyword fista_p: (float)
        Default: 1 for standard FISTA 
        0.01 < fista_p <= 0.1 for faster FISTA
        
    :keyword fista_q: (float)
        Default: 1 for standard FISTA 
        0.0 < fista_q <= 1.0 for faster FISTA

    Usage
    --------
    >>> import numpy as np
    >>> import tigre
    >>> import tigre.algorithms as algs
    >>> from tigre.demos.Test_data import data_loader
    >>> geo = tigre.geometry(mode='cone',default_geo=True,
    >>>                         nVoxel=np.array([512,512,512]))
    >>> angles = np.linspace(0,2*np.pi,100)
    >>> src_img = data_loader.load_head_phantom(geo.nVoxel)
    >>> proj = tigre.Ax(src_img,geo,angles)
    >>> output = algs.fista(proj,geo,angles,niter=50
    >>>                                 hyper=2.e4)

    tigre.demos.run() to launch ipython notebook file with examples.


    --------------------------------------------------------------------
    This file is part of the TIGRE Toolbox

    Copyright (c) 2015, University of Bath and
                        CERN-European Organization for Nuclear Research
                        All rights reserved.

    License:            Open Source under BSD.
                        See the full license at
                        https://github.com/CERN/TIGRE/license.txt

    Contact:            tigre.toolbox@gmail.com
    Codes:              https://github.com/CERN/TIGRE/
    --------------------------------------------------------------------
    Coded by:          MATLAB (original code): Ander Biguri
                       PYTHON : Reuben Lindroos

    """

    def __init__(self, proj, geo, angles, niter, **kwargs):
        # Don't precompute W and V
        kwargs.update({"W": None, "V": None})
        kwargs.update(dict(blocksize=angles.shape[0]))
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)
        self.lmbda = 0.1
        # hyper = the Lipschitz constant L of the data term (largest
        # eigenvalue of A^T A); the gradient step is 1/L. It is PROBLEM-SIZE
        # dependent (the legacy 2.0e8 constant is calibrated to a 64^3 demo;
        # the docstring above quotes ~2e4 at 512^3), so a hardcoded default
        # makes FISTA's steps orders of magnitude too small on production
        # geometry - the image just stays near its initialization. Default is
        # now "auto": estimate L by power iteration on THIS problem's actual
        # operators. Pass a number to reproduce the old behaviour.
        self.__L__ = kwargs.get("hyper", "auto")
        self.__numiter_tv__ = 20 if "tviter" not in kwargs else kwargs["tviter"]
        self.__lambda__ = 0.1 if "tvlambda" not in kwargs else kwargs["tvlambda"]
        self.__t__ = 1
        if isinstance(self.__L__, str):
            self.__L__ = self._estimate_lipschitz()
        self.__bm__ = 1.0 / self.__L__
        self.__p__ = 1 if "fista_p" not in kwargs else kwargs["fista_p"]
        self.__q__ = 1 if "fista_q" not in kwargs else kwargs["fista_q"]

    def _estimate_lipschitz(self, n_iter=8, seed=0):
        """Estimate L = lambda_max(A^T A) by power iteration, using the SAME
        operator pair as update_image (Ax "interpolated" / Atb "matched") so
        the estimate matches the gradient actually taken. Cost: n_iter
        forward+back projection pairs - comparable to n_iter FISTA iterations,
        paid once.

        Scaling: update_image applies an effective step of 1/hyper to the
        LEAST-SQUARES gradient grad f = 2 A^T(Ax - b), whose Lipschitz
        constant is 2*lambda_max(A^T A). Power iteration measures
        lambda_max(A^T A), so hyper must be at least TWICE that - returning
        the bare estimate (x1.05) was verified to diverge to NaN on a
        synthetic phantom. The extra 1.05 biases the (from-below-converging)
        power estimate to the safe side: overestimating hyper only shrinks
        the step slightly, underestimating it diverges."""
        rng = np.random.default_rng(seed)   # deterministic: repeat runs match
        x = rng.standard_normal(tuple(self.geo.nVoxel)).astype(np.float32)
        x /= np.linalg.norm(x.ravel())
        L = 1.0
        for _ in range(n_iter):
            y = tigre.Atb(
                tigre.Ax(x, self.geo, self.angles, "interpolated", gpuids=self.gpuids),
                self.geo, self.angles, "matched", gpuids=self.gpuids)
            L = float(np.linalg.norm(y.ravel()))
            x = y / L
        L *= 2.0 * 1.05
        if self.verbose:
            print(f"FISTA: estimated Lipschitz constant (hyper) = {L:.4g} "
                  f"(2 x lambda_max(A^T A) from {n_iter} power iterations)")
        return L

    # override update_image from iterative recon alg to remove W.
    def update_image(self, geo, angle, iteration):
        """
        VERBOSE:
         for j in range(angleblocks):
             angle = np.array([alpha[j]], dtype=np.float32)
             proj_err = proj[angle_index[j]] - Ax(res, geo, angle, 'Siddon')
             backprj = Atb(proj_err, geo, angle, 'FDK')
             res += backprj
             res[res<0]=0

        :return: None
        """
        self.res += (
            self.__bm__
            * 2
            * tigre.Atb(
                (
                    self.proj[self.angle_index[iteration]]
                    - tigre.Ax(self.res, geo, angle, "interpolated", gpuids=self.gpuids)
                ),
                geo,
                angle,
                "matched",
                gpuids=self.gpuids,
            )
        )

    def run_main_iter(self):
        """
        Goes through the main iteration for the given configuration.
        :return: None
        """
        t = self.__t__
        Quameasopts = self.Quameasopts
        x_rec = copy.deepcopy(self.res)
        lambdaForTv = 2 * self.__bm__ * self.__lambda__
        for i in range(self.niter):

            res_prev = copy.deepcopy(self.res) if Quameasopts is not None else None
            if self.verbose:
                self._estimate_time_until_completion(i)

            getattr(self, self.dataminimizing)()

            x_rec_old = copy.deepcopy(x_rec)
            x_rec = im3ddenoise(self.res, self.__numiter_tv__, 1.0 / lambdaForTv, self.gpuids)
            t_old = t
            # float(): np.sqrt returns an np.float64 SCALAR, and under NumPy 2
            # promotion (NEP 50) a np.float64 scalar UPCASTS a float32 array -
            # the momentum line below then silently turned self.res float64 and
            # the next iteration's Ax raised "Input data should be float32".
            # A python float is a weak scalar and leaves the array dtype alone.
            t = float((self.__p__ + np.sqrt(self.__q__ + 4 * t ** 2)) / 2)
            self.res = x_rec + (t_old - 1) / t * (x_rec - x_rec_old)
            
            if Quameasopts is not None:
                self.error_measurement(res_prev, i)


fista = decorator(FISTA, name="FISTA")


class ISTA(FISTA):  # noqa: D101
    __doc__ = FISTA.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        FISTA.__init__(self, proj, geo, angles, niter, **kwargs)

    def run_main_iter(self):
        """
        Goes through the main iteration for the given configuration.
        :return: None
        """
        Quameasopts = self.Quameasopts
        lambdaForTv = 2 * self.__bm__ * self.lmbda
        for i in range(self.niter):

            res_prev = copy.deepcopy(self.res) if Quameasopts is not None else None
            if self.verbose:
                self._estimate_time_until_completion(i)

            getattr(self, self.dataminimizing)()

            self.res = im3ddenoise(self.res, 20, 1.0 / lambdaForTv, self.gpuids)
            
            if Quameasopts is not None:
                self.error_measurement(res_prev, i)


ista = decorator(ISTA, name="ISTA")
