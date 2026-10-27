# %% Demo 20: FISTA with a TV proximal operator
#
#
#  This demo presents the FISTA algorithm in TIGRE. using a total variation proximal.
#  Total variation algorithms try to minimize the variation (gradient) of the
#  image, assuming its piecewise smooth, as most things in nature are (i.e.
#  human body). FISTA attempts to perform this with much faster convergence
#  than standard gradient algorithms such as SIRT.
#
#
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This file is part of the TIGRE Toolbox
#
# Copyright (c) 2015, University of Bath and
#                     CERN-European Organization for Nuclear Research
#                     All rights reserved.
#
# License:            Open Source under BSD.
#                     See the full license at
#                     https://github.com/CERN/TIGRE/blob/master/LICENSE
#
# Contact:            tigre.toolbox@gmail.com
# Codes:              https://github.com/CERN/TIGRE/
# Coded by:           Ander Biguri, Reuben Lindroos
# --------------------------------------------------------------------------
#%%Initialize
import tigre
import numpy as np
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from matplotlib import pyplot as plt

#%% Geometry
geo = tigre.geometry_default(high_resolution=False)
# Match the MATLAB acquisition; Python volume and detector axes are reversed.
geo.nVoxel = np.array([128, 128, 128])
geo.sVoxel = np.array([256.0, 256.0, 256.0])
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([256, 256])
geo.dDetector = np.array([409.6, 409.6]) / geo.nDetector
geo.sDetector = geo.nDetector * geo.dDetector
geo.accuracy = 1.0

#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
# Load bundled head phantom data
head = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)
# add noise
noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))


## FISTA is a quadratically converging algorithm.

# 'hyper': Gradient step denominator, estimated with the native operator
#          when omitted. Explicit values below are illustrative and depend
#          on the geometry. Python uses tvlambda where MATLAB uses lambda.
# 'tviter':  Number of iterations of Im3ddenoise to use. Default: 20
# 'lambda':  Multiplier for the tvlambda used, which is proportional to
#            L (hyper). Default: 0.1
# 'verbose': get feedback or not. Default: 1
# --------------------------------------------------------------------------
imgFISTA_default = algs.fista(noise_projections, geo, angles, 100)

## Adding a different hyper parameter
imgFISTA_hyper = algs.fista(noise_projections, geo, angles, 100, hyper=2.0e6)

## Recon more significant tv parameters
imgFISTA_hightv = algs.fista(
    noise_projections, geo, angles, 100, hyper=2.0e6, tviter=100, tvlambda=20
)

imgFISTA_mod = algs.fista(
    noise_projections, geo, angles, 100, hyper=1.0e5,
    tviter=100, tvlambda=20, fista_q=1/50, fista_p=1/20,
)


## Plot results
tigre.plotimg(
    np.concatenate([imgFISTA_default, imgFISTA_hyper, imgFISTA_hightv, imgFISTA_mod], axis=2),
    dim="z",
    clims=[0, 1],
)
