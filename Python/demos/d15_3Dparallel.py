# %% Demo 15: 3D parallel tomography
#
#
# This demo shows how to run 3D parallel geometry
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
# While you can define your own geometry, the easiest way is:

# just give as nVoxel the same number as your projection size.
geo = tigre.geometry(
    mode="parallel", nVoxel=np.array([128, 128, 128])
)  # Parallel beam geometry does not require anything other than the image size.
geo.DSD, geo.DSO = 2000, 1000
geo.sVoxel = np.array([512.0, 512.0, 512.0])
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([128, 128])
geo.dDetector = np.array([5.5, 5.5])
geo.sDetector = geo.nDetector * geo.dDetector
geo.accuracy = 0.5


#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2 * np.pi, 360)
# Load bundled head phantom data.
head = sample_loader.load_head_phantom(geo.nVoxel)
# generate projections
projections = tigre.Ax(head, geo, angles)
# add noise
noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))

# recon
imgFBP = algs.fbp(projections, geo, angles)
imgOSSART = algs.os_sart(projections, geo, angles, 40)

# plot
tigre.plotImg(np.concatenate([head, imgFBP, imgOSSART], axis=1), dim="z", step=3)
