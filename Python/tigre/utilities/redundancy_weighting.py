import numpy as np
import copy
import cupy as cp

    
def apply_redudancy_weights(geo, verbose=True) -> bool:
    """
    Check if Wang redundancy weights should be applied

    Parameters
    ----------
    geo : Object
        The Tigre geometry 

    Returns
    -------
    True if applicable

    """
    if (np.atleast_2d(geo.offDetector).shape[0] > 1) and np.atleast_2d(geo.offDetector)[:, 1].ptp() > 0:
        if verbose:
            print('Wang weights: varying offDetector detected, Wang weights not being applied')
        return False
    
    if np.atleast_2d(geo.offDetector)[0, 1] == 0: 
        if verbose:
            print('0 Detector offset, no Wang weights needed')
        return False
    
    if (len(np.atleast_2d(geo.DSO)) > 1) and len(np.unique(geo.DSO)) > 1:
        if verbose:
            print('Wang weights: varying DSO detected, Wang weights not being applied');
        return False

    percent_offset = abs(np.atleast_2d(geo.offDetector)[0, 1] / np.atleast_2d(geo.sDetector)[0, 1]) * 100;    
    if percent_offset > 30:
        print('Wang weights: Detector offset percent: %0.2f is greater than 30 which may result in image artifacts, consider rebinning 360 degree projections to 180 degrees', percent_offset)
    
    return True


def redundancy_weighting(geo):
    """
    Preweighting using Wang function
    Ref: 
        Wang, Ge. X-ray micro-CT with a displaced detector array. Medical Physics, 2002,29(7):1634-1636.
    """

    if not hasattr(geo,'COR'):
        geo.COR=np.array([0])
    
    w = np.ones((geo.nDetector[0], geo.nDetector[1]), dtype=np.float32)
    
    if apply_redudancy_weights(geo):
        offset = np.atleast_2d(geo.offDetector)[0, 1]
        DSD = np.atleast_2d(geo.DSD)[0]
        DSO = np.atleast_2d(geo.DSO)[0]
        offset += (DSD / DSO) * np.atleast_1d(geo.COR)[0]   # added correction
        us = np.linspace(-geo.nDetector[1]/2+0.5, geo.nDetector[1]/2-0.5, geo.nDetector[1]) * geo.dDetector[1] + abs(offset)
        
        us *= DSO / DSD
        theta = (geo.sDetector[1]/2 - abs(offset)) * np.sign(offset)
        abstheta = abs(theta * DSO / DSD)
    
        w = np.where(np.abs(us) <= abstheta,
            0.5 * (np.sin( (np.pi / 2) * np.arctan(us / DSO) / np.arctan(abstheta / DSO) ) + 1),
            np.where(us < -abstheta, 0, w)
            )
#        w=w*2
        if (theta<0):
            w = np.fliplr(w)
            
    return w.astype(np.float32)


def zeropadding(proj, geo):
    """
    Zero padding the projections and modify geometry accordingly

    Parameters
    ----------
    proj : ndarray or cupy array
        Projections.
    geo : obj
        geometry for reconstruction.

    Returns
    -------
    zproj : ndarray or cupy array
        Zero padding the projections.
    zgeo : obj
        modified geometry for reconstruction.
    theta : ndarray
        Angles of projection.

    """
    xp = cp.get_array_module(proj)
    zgeo = copy.copy(geo)
    
    offDet1 = np.atleast_2d(geo.offDetector)[0, 1]
    offDet1 += np.atleast_1d(geo.DSD)[0] / np.atleast_1d(geo.DSO)[0] * np.atleast_1d(geo.COR)[0]
    
    width = int(np.fix(2 * offDet1 / geo.dDetector[1])) + 1
    if np.isscalar(geo.DSO):
        zgeo.offDetector[1] = zgeo.offDetector[1] - width / 2 * geo.dDetector[1]
    else:
        zgeo.offDetector[:, 1] = zgeo.offDetector[:, 1] - width / 2 * geo.dDetector[1]

    zgeo.nDetector[1] += abs(width)
    zgeo.sDetector[1] = zgeo.nDetector[1] * zgeo.dDetector[1]

    padwidth = ((0, 0), (0, 0), (width, 0)) if offDet1 > 0 \
        else ((0, 0), (0, 0), (0, abs(width)))
    zproj = xp.pad(proj, padwidth, constant_values=0)
        
    return zproj, zgeo
