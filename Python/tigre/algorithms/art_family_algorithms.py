import copy

import numpy as np

from tigre.algorithms.iterative_recon_alg import IterativeReconAlg
from tigre.algorithms.iterative_recon_alg import decorator
from tigre.utilities.im_3d_denoise import im3ddenoise



class SART(IterativeReconAlg):  
    __doc__ = (
        "SART solves Cone Beam CT image reconstruction using \n"
        "Simultaneous Algebraic Reconstruction Technique algorithm\n"
        "SART(PROJ,GEO,ALPHA,NITER) solves the reconstruction problem\n"
        "using the projection data PROJ taken over ALPHA angles, corresponding\n"
        "to the geometry described in GEO, using NITER iterations. \n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        if "blocksize" in kwargs and kwargs['blocksize']>1:
            print('Warning: blocksize is set to 1, please use an OS version of the algorithm for blocksize > 1')
        kwargs.update(dict(blocksize=1))
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)


sart = decorator(SART, name="sart")


class SIRT(IterativeReconAlg):  
    __doc__ = (
        "SIRT solves Cone Beam CT image reconstruction using \n"
        "Simultaneous Iterative Reconstructive Technique algorithm\n"
        "SIRT(PROJ,GEO,ALPHA,NITER) solves the reconstruction problem\n"
        "using the projection data PROJ taken over ALPHA angles, corresponding\n"
        "to the geometry described in GEO, using NITER iterations.\n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        if "blocksize" in kwargs and kwargs['blocksize']>1:
            print('Warning: blocksize is set to {}, please do not specify blocksize for this algorithm'.format(angles.shape[0]))
        kwargs.update(dict(blocksize=angles.shape[0]))
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)


sirt = decorator(SIRT, name="sirt")


class OS_SART(IterativeReconAlg):  
    __doc__ = (
        "OS_SART solves Cone Beam CT image reconstruction using Oriented Subsets\n"
        "Simultaneous Algebraic Reconstruction Technique algorithm\n"
        "OS_SART(PROJ,GEO,ALPHA,NITER,BLOCKSIZE=20) solves the reconstruction problem\n"
        "using the projection data PROJ taken over ALPHA angles, corresponding\n"
        "to the geometry described in GEO, using NITER iterations.\n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        
        self.blocksize = 20 if 'blocksize' not in kwargs else kwargs["blocksize"]       
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)


os_sart = decorator(OS_SART, name="os_sart")


class Fast_OS_SART(IterativeReconAlg):  # noqa: N801
    __doc__ = (
        "FAST_OS_SART is OS_SART with Nesterov/FISTA momentum: before each\n"
        "pass over the subsets, the iterate is extrapolated along the last\n"
        "step, y_k = x_k + (t_{k-1}-1)/t_k * (x_k - x_{k-1}), with the\n"
        "classical t schedule t_k = (1+sqrt(1+4 t_{k-1}^2))/2.\n"
        "After CERN/TIGRE PR #751 (open, unmerged as of 17 Aug 2026).\n"
        "CAUTION: momentum over ordered subsets is an acceleration heuristic,\n"
        "not a convergent algorithm -- with noisy/inconsistent data it also\n"
        "accelerates semi-convergence, so it reaches the noise-fitting regime\n"
        "in fewer iterations too. Compare against OS_SART on the\n"
        "sharpness/noise FRONTIER, never at matched iteration counts.\n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        self.blocksize = 20 if 'blocksize' not in kwargs else kwargs["blocksize"]
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)

    def run_main_iter(self):
        t = 1.0
        x_old = self.res.copy()
        for i in range(self.niter):
            res_prev = None
            if self.Quameasopts is not None:
                res_prev = copy.deepcopy(self.res)
            if self.verbose:
                self._estimate_time_until_completion(i)
            t_old, t = t, (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            x_cur = self.res.copy()
            # extrapolate, then let the data step descend IN PLACE from y
            self.res += ((t_old - 1.0) / t) * (self.res - x_old)
            x_old = x_cur
            getattr(self, self.dataminimizing)()
            self.error_measurement(res_prev, i)


fast_os_sart = decorator(Fast_OS_SART, name="fast_os_sart")


class SART_TV(IterativeReconAlg):  
    __doc__ = (
        "SART_TV solves Cone Beam CT image reconstruction using Simultaneous \n"
        "Algebraic Reconstruction Technique with TV regularization algorithm\n"
        "SART_TV(PROJ,GEO,ALPHA,NITER,TVLAMBDA=50,TVITER=50) solves the reconstruction\n"
        "problem using the projection data PROJ taken over ALPHA angles\n"
        "corresponding to the geometry described in GEO, using NITER iterations. \n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        
        if "blocksize" in kwargs and kwargs['blocksize']>1:
            print('Warning: blocksize is set to 1, please use an OS version of the algorithm for blocksize > 1')
        kwargs.update(dict(blocksize=1))
        self.tvlambda = 50 if 'tvlambda' not in kwargs else kwargs['tvlambda']
        self.tviter = 50 if 'tviter' not in kwargs else kwargs['tviter']
        # these two settings work well for nVoxel=[254,254,254]

        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)

    # Override
    def run_main_iter(self):
        """
        Goes through the main iteration for the given configuration.
        :return: None
        """
        Quameasopts = self.Quameasopts

        for i in range(self.niter):

            res_prev = None
            if Quameasopts is not None:
                res_prev = copy.deepcopy(self.res)
            if self.verbose:
                self._estimate_time_until_completion(i)

            getattr(self, self.dataminimizing)()
            # print("run_main_iter: gpuids = {}", self.gpuids)
            self.res = im3ddenoise(self.res, self.tviter, self.tvlambda, self.gpuids)
            if Quameasopts is not None:
                self.error_measurement(res_prev, i)


sart_tv = decorator(SART_TV, name="sart_tv")


class OS_SART_TV(IterativeReconAlg):  
    __doc__ = (
        "OS_SART_TV solves Cone Beam CT image reconstruction using Oriented Subsets\n"
        "Simultaneous Algebraic Reconstruction Technique with TV regularization algorithm\n"
        "OSSART_TV(PROJ,GEO,ALPHA,NITER,BLOCKSIZE=20,TVLAMBDA=50,TVITER=50) \n"
        "solves the reconstruction problem using the projection data PROJ taken\n"
        "over ALPHA angles, corresponding to the geometry described in GEO,\n"
        "using NITER iterations.\n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        
        self.blocksize = 20 if 'blocksize' not in kwargs else kwargs['blocksize']
        self.tvlambda = 50 if 'tvlambda' not in kwargs else kwargs['tvlambda']
        self.tviter = 50 if 'tviter' not in kwargs else kwargs['tviter']
        # these two settings work well for nVoxel=[254,254,254]

        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)

   # Override
    def run_main_iter(self):
        """
        Goes through the main iteration for the given configuration.
        :return: None
        """
        Quameasopts = self.Quameasopts

        for i in range(self.niter):

            res_prev = None
            if Quameasopts is not None:
                res_prev = copy.deepcopy(self.res)
            if self.verbose:
                self._estimate_time_until_completion(i)
            
            getattr(self, self.dataminimizing)()
            # print("run_main_iter: gpuids = {}", self.gpuids)
            self.res = im3ddenoise(self.res, self.tviter, self.tvlambda, self.gpuids)
            if Quameasopts is not None:
                self.error_measurement(res_prev, i)

os_sart_tv = decorator(OS_SART_TV, name="os_sart_tv")
