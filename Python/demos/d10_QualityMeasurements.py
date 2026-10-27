# %% Demo 10: Image quality measurements
#
# This demo demonstrate how to compute all the image quality measures
# by calling the "Measure_Quality.py" function with detailed description.
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
from tigre.utilities.Measure_Quality import Measure_Quality as MQ
from tigre.utilities.visualization.plot_quality_measures import plot_quality_measures

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


#%% Image quality measurement
# There are three inputs to the function

#
#           'RMSE' : The square root of the mean of the squared differences
#                    of pixel intensities of two images.
#
#                    Taken from "A rigid motion correction method for helical
#                    computed tomography (CT)"
#                    (doi:10.1088/0031-9155/60/5/2047)
#           'nRMSE': The square root of the mean of the squared differences
#                    of pixel intensities of two images, normalized
#
#           'CC'   : The Pearson correlation coefficient which measures the
#                    linear dependence between two images.
#
#                    Taken from "A rigid motion correction method for helical
#                    computed tomography (CT)"
#                    (doi:10.1088/0031-9155/60/5/2047)
#
#           'MSSIM': The mean structural similarity index, a measure of the
#                    similarity of two images in terms of luminance,
#                    contrast and structure that is designed to provide a
#                    good approximation of perceptual image quality.
#
#                    Taken from "A rigid motion correction method for helical
#                    computed tomography (CT)"
#                    (doi:10.1088/0031-9155/60/5/2047)
#
#           'UQI' : The universal quality index to evaluate the degree of
#                   similarity between the reconstructed and phantom images
#                   for chosen ROIs. Its value ranges from zero to one.
#                   A UQI value closer to one suggests better similarity to true image.
#
#                   Taken from "Few-view cone-beam CT reconstruction with
#                   deformed prior image" (doi: 10.1118/1.4901265)
#
#
#
#

## Define a cell array of image quality measures
qualmeas = ["RMSE", "CC", "MSSIM", "UQI"]


#%% Reconstruct image using SIRT, SART and OS-SART
# =========================================================================
imgSIRT, qualitySIRT = algs.sirt(noise_projections, geo, angles, 30, Quameasopts=qualmeas)
imgSART, qualitySART = algs.sart(noise_projections, geo, angles, 30, Quameasopts=qualmeas)
imgOSSART, qualityOSSART = algs.os_sart(noise_projections, geo, angles, 30, blocksize=22, Quameasopts=qualmeas)

#%% Plot the changes of image quality measures over iterations, i.e. how much each iteration looks like the previous iteration.
# (without the original image one can not compute the absolute image quality, of course)

# Plot each algorithm on its own completed-iteration axis, without stacking
# or padding histories. The ART solvers currently run all requested iterations.
plot_quality_measures(
    {"SIRT": qualitySIRT, "SART": qualitySART, "OS-SART": qualityOSSART}, qualmeas)

plt.show()

#%% Compute the parameters for the target image
# Knowing change of the parameters per iteration can be interesting, but
# The main use of them would be to compare a reconstructed image to a given
# known target image.

sart_uqi = MQ(head, imgSART, ["UQI"])
sirt_uqi = MQ(head, imgSIRT, ["UQI"])
ossart_uqi = MQ(head, imgOSSART, ["UQI"])

print("UQI for SART,   SIRT,   OS-SART")
print("       " + str(sart_uqi) + " " + str(sirt_uqi) + " " + str(ossart_uqi))

sart_rmse = MQ(head, imgSART, ["RMSE"])
sirt_rmse = MQ(head, imgSIRT, ["RMSE"])
ossart_rmse = MQ(head, imgOSSART, ["RMSE"])

print("RMSE for SART,   SIRT,   OS-SART")
print("       " + str(sart_rmse) + " " + str(sirt_rmse) + " " + str(ossart_rmse))
