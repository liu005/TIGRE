# %% Demo 11: Image postprocessing
#
# This demo demonstrates the available postprocessing tools in TIGRE.by calling the "Measure_Quality.m" function with detailed description.
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
# Coded by:           Manasavee Lohvithee
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
angles = np.linspace(0, 2 * np.pi, 100)
# Load bundled head phantom data
head = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)
# add noise
noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))

#%% Some recon, FDK for example
imgFDK = algs.fdk(noise_projections, geo, angles)

from tigre.utilities.im_3d_denoise import im3ddenoise

# Match MATLAB's TV denoising example. Cone-support cropping remains MATLAB-only.
imgdenoised = im3ddenoise(imgFDK.copy(), iter=100, lmbda=15)
tigre.plotimg(np.concatenate([imgFDK, imgdenoised], axis=2), dim="z")
tigre.plotimg(np.concatenate([abs(head-imgFDK), abs(head-imgdenoised)], axis=2), dim="z")
