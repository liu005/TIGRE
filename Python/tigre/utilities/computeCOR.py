# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 22:03:35 2025

@author: liu005
"""

import numpy as np
import cupy as cp
from cv2 import remap
#from scipy.interpolate import griddata

def computeCOR(data, geo, angles, slc=None, gpu=True):
    # Code modified from SophiaBeads
    # (https://github.com/Sophilyplum/sophiabeads-datasets/blob/master/tools/centre_geom.m)
    # Reference:
    # T. Liu - "Direct central ray determination in computed microtomography",
    # Optical Engineering, April 2009.
    
    # Get central slice
    if slc is None:
        slc = int(np.floor(data.shape[1]/2))#+1
        
    data=cp.squeeze(data[:,slc,:])
    
    if gpu:
        from cupyx.scipy.interpolate import RegularGridInterpolator
        xp = cp
        data = cp.asarray(data)
        angles = cp.asarray(angles)
    else:
        from scipy.interpolate import RegularGridInterpolator
        xp = np
    
    # if size(angles,1)==1
    #    angles=angles'; 
    # end
    
    # Set up coordinate grids for testing the fit to data
    angle_grid, det_grid = xp.meshgrid(angles, xp.linspace(-geo.sDetector[1]/2+geo.dDetector[1]/2,+geo.sDetector[1]/2-geo.dDetector[1]/2,geo.nDetector[1]), indexing='ij')
    angle_grid = xp.vstack( (angle_grid - 2*xp.pi, angle_grid, angle_grid + 2*xp.pi) )
    det_grid = xp.vstack( (det_grid, det_grid, det_grid) )
    test_data = xp.vstack( (data, data, data) )
    
#    test_data = np.float64(repmat(data, 3, 1))
    # points = xp.column_stack((angle_grid.flatten(), det_grid.flatten()))
    # interp2d_lin = RegularGridInterpolator(points, test_data)
    # Start search using midpoint at zero
    midpoint = 0
    # Vector of precision values to search at
    precision = xp.array([1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    
    for pr in precision:
        
#        COR = midpoint - np.linspace(10*pr, midpoint + 10*pr, pr) # values for centre of rotation
        COR = xp.linspace(midpoint - 10*pr, midpoint + 10*pr, 21) # values for centre of rotation
        M = xp.zeros((len(COR),))
        
        for j in range(len(COR)):
            
            gamma = np.arctan(xp.linspace(-geo.sDetector[1]/2+geo.dDetector[1]/2,+geo.sDetector[1]/2-geo.dDetector[1]/2,geo.nDetector[1]) / geo.DSD)   # angle of each ray relative to theoretical central ray    
            gamma_c = xp.arctan(COR[j] / geo.DSD) # angle of assumed centre of rotation to central ray
            gamma_i = gamma - gamma_c
            beta = 2 * gamma_i + xp.pi
            
            s2 = geo.DSD * xp.tan(2 * gamma_c - gamma)
            s2 = xp.tile(s2, (angles.shape[0], 1))
            
            angles_aux = xp.tile(angles, (geo.nDetector[1], 1)).T + xp.tile(beta, (angles.shape[0], 1))
            test = remap( test_data, angle_grid, (angles_aux, s2))
#            test = interpn((angle_grid, det_grid), test_data, (angles_aux, s2), 'linear');
            
            nonzero = (test > 0)
            # We want the number of non-zero values for the average, not the sum of their positions
            M[j] = sum((test[nonzero] - data[nonzero])**2)*(1/len(nonzero))
        
        indM = xp.argmin(M)   # minimum value and index
        midpoint = COR[indM]   # set midpoint for next search
    
    # transform centre to required value
    centre =- midpoint * geo.DSO / geo.DSD
    return centre.get() if gpu else centre
