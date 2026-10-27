# %% Demo 04: Simple FDK and OS-SART reconstruction
#
#
# This demo will show how a simple image reconstruction can be performed,
# by using OS-SART and FDK
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
# Coded by:           Ander Biguri
# --------------------------------------------------------------------------
#%%Initialize
import tigre
import numpy as np
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs

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
angles = np.linspace(0, 2 * np.pi, 100)
# Load bundled head phantom data.
head = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)
# add noise
noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))

#%% Reconstruct image using OS-SART and FDK

# FDK
imgFDK = algs.fdk(noise_projections, geo, angles)
# OS-SART

niter = 50
imgOSSART = algs.os_sart(noise_projections, geo, angles, niter)

#%% Show the results
tigre.plotimg(np.concatenate([imgFDK, imgOSSART], axis=1), dim="z")
