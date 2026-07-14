
import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import animation
import tigre

   
def animate_geometry(ogeo,angles=None,pos=None,rotation="SD",animate=True,fname=None):
    """
    Plot animated TIGRE geometry

    Parameters
    ----------
    ogeo : tigre geometry object
        Geometry to be plotted.
    angles : array, optional
        Angles to plot for the geometry. The default is np.linspace(0,2*np.pi,100).
    pos : float, optional
        A specific angles[pos] for static geometry plot. The default is 0, 
        angles[0] is used.
    rotation : string, optional
        Style of animated plot with rotation style: "SD" - fixed object, source 
        and detector moving; "obj" - source and detector fixed, object moves, 
        only applicable when source and detector are not changing relative positions. 
        The default is "obj".
    animate : bool, optional
        Plot animated geometry setup or static geometry. The default is True.
    fname : string, optional
        Filename to save animated geometry plotting. The default is None.

    Returns
    -------
    Handle for animation or plot.

    """
#PLOT_GEOMETRY(GEO,ANGLES,POS,Animate,fname) plots a simplified version of the CBCT geometry with the
# given geomerty GEO and scanning angles ANGLES at angle POS. If angles is 
# not given, [0,2pi] at 100 steps will be given, and pos=0 will be chosen. If 
# Animate=TRUE, an animation file will be generated. Animation file name can be
# specified by fname.
# 
# h=PLOT_GEOMETRY(...) will return the figure handle 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
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
# Coded by:           Ander Biguri, modified by Yi Liu
#--------------------------------------------------------------------------

    # check geo, which makes DSD, DSO etc all matrices
    if angles is None:
        angles = ogeo.angles if hasattr(ogeo,"angles") else np.linspace(0,2*np.pi,100)
        
    geo = copy.deepcopy(ogeo)
    try:
        geo.check_geo(angles)
    except:
        raise ValueError("Geometry and angles are inconsistent. Please check inputs.")
        
    # Validate pos index
    pos = 0 if pos is None or pos < 0 or pos >= len(angles) else pos
            
    # Set up figure 
    fig=plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=52,elev=26)       
    ln = geo.sVoxel.min()/2   

    # Define rotation matrices
    Rs = Rotation.from_euler('ZYZ', geo.angles).as_matrix()
    
    # Compute source and detector centres
    scent = np.column_stack((geo.DSO, np.zeros_like(geo.angles[:, 1]), np.zeros_like(geo.angles[:, 2])))
    dcent = np.column_stack((-geo.DSD+geo.DSO, geo.offDetector[:,1], geo.offDetector[:,0]))
   
    def plot_trajectory(points, color, label=None, marker_size=2.5):
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], color=color, ls='', marker=".", markersize=marker_size, mfc=color, mec=color)
        if label == None:
            return ax.scatter(points[pos, 0], points[pos, 1], points[pos, 2], color=color, s=5)
        else:
            return ax.scatter(points[pos, 0], points[pos, 1], points[pos, 2], color=color, s=5), ax.text(points[pos, 0], points[pos, 1], points[pos, 2] + 30, label)

    # Compute source trajectory, coordinates in (z,y,x)
    if rotation == "SD":
        stj = np.einsum('ijk,ik->ij', Rs, scent)
        if hasattr(geo, 'COR'):
            stj[:, 1] += geo.COR.flatten()
    elif rotation == "obj":
        stj = np.zeros_like(geo.angles)
        stj[pos,:] = np.matmul(Rs[pos,:,:], scent[pos,:])
        if hasattr(geo, 'COR'):
            stj[pos,1] += geo.COR[pos]
    
    source, stext = plot_trajectory(stj, 'plum', 'S')
        
    # Compute detector trajectory
    if rotation == "SD":
        dtj = np.einsum('ijk,ik->ij', Rs, dcent)
        if hasattr(geo, 'COR'):
            dtj[:, 1] += geo.COR.flatten()
    elif rotation == "obj":
        dtj = np.zeros_like(geo.angles)
        dtj[pos,:] = np.matmul(Rs[pos,:,:], dcent[pos,:]) 
        if hasattr(geo, 'COR'):
            dtj[:,1] += geo.COR.flatten()

    det, dtext = plot_trajectory(dtj, 'skyblue', 'D')
                
    # Compute detector orientation
    R_detector = np.matmul(Rs, Rotation.from_euler('XYZ', geo.rotDetector).as_matrix())
    
    # Detector cube, cp returns four cordinates of corners closest to source
    ddp = 30 # detector depth
    dsz = np.array([ddp, geo.sDetector[1], geo.sDetector[0]])  # at angles (0,0,0)
    dverts = calCube(dtj, dsz, R_detector, offcent=np.array([-ddp / 2, 0, 0]))
    dcube = art3d.Poly3DCollection(dverts[pos], color='brown', alpha=0.2)
    ax.add_collection3d(dcube)
        
    
    # Compute image origin trajectory. NOTE: offOrigin is in (z,y,x)
    otj = np.fliplr(geo.offOrigin).astype(np.float32)
    # # displacement in y for geo.COR
    if hasattr(geo, 'COR'):
        otj[:,1] += geo.COR.flatten()   
    _ = plot_trajectory(otj, 'grey')    
    
    # Compute object cube
    overts = calCube(otj, geo.sVoxel[[2, 1, 0]], np.swapaxes(Rs, 1, 2) if rotation == "obj" else 'eye')
    ocube = art3d.Poly3DCollection(overts[pos], color='c', alpha=0.2)
    ax.add_collection3d(ocube)    
    
    # Cordinates Arrows from origin
    ax.quiver(0, 0, 0, 1, 0, 0, length=ln, color='r')
    ax.quiver(0, 0, 0, 0, 1, 0, length=ln, color='b')
    ax.quiver(0, 0, 0, 0, 0, 1, length=ln, color='g')
    ax.text(-10, -10, -10, 'O', None)
        
    # Effective beam profile
    beampf = [ax.plot3D(*zip(stj[pos, :], dverts[pos][0][i]), color='y') for i in range(4)]

    # Cenrtal beam
    cbeam=ax.plot3D(*zip(stj[pos,:],dtj[pos,:]),color='pink')
    
    # set tight limits and aspect
    lims = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    ax.set_box_aspect([lim[1] - lim[0] for lim in lims])

    # Set up plot title and axis label
    roll, pitch, yaw = geo.rotDetector[:,0]/np.pi*180, geo.rotDetector[:,1]/np.pi*180, geo.rotDetector[:,2]/np.pi*180
    if rotation == 'SD':
        ang = geo.angles[pos] / np.pi * 180
        title = ax.set_title(
            f"CBCT geometry in scale - Source at angle [{ang[0]:.1f}, {ang[1]:.1f}, {ang[2]:.1f}]°\n\n"
            f"Detector rotation [{roll[pos]:.1f}, {pitch[pos]:.1f}, {yaw[pos]:.1f}]°\n\n"
            f"Detector offset [{geo.offDetector[pos, 0]:.1f}, {geo.offDetector[pos, 1]:.1f}] mm")
    elif rotation == 'obj':
        ang = -geo.angles[pos] / np.pi * 180
        title = ax.set_title(
            f"CBCT geometry in scale - Object at angle [{ang[0]:.1f}, {ang[1]:.1f}, {ang[2]:.1f}]°\n\n"
            f"Detector rotation [{roll[pos]:.1f}, {pitch[pos]:.1f}, {yaw[pos]:.1f}]°\n\n"
            f"Detector offset [{geo.offDetector[pos, 0]:.1f}, {geo.offDetector[pos, 1]:.1f}] mm")
    ax.set_xlabel('X');
    ax.set_ylabel('Y');
    ax.set_zlabel('Z');
          
    # Define update function
    def update(pos):
        ocube.set_verts(overts[pos]) # to move or rotate
        if rotation == 'SD':
            source.set_offsets(stj[pos, :2])
            source.set_3d_properties(stj[pos, 2], 'z')
            stext.set_position(stj[pos,:2] + 15)
            det.set_offsets(dtj[pos, :2])
            det.set_3d_properties(dtj[pos, 2], 'z')
            dtext.set_position(dtj[pos, :2] + 15)
            cbeam[0].set_data(*zip(stj[pos, :2], dtj[pos, :2]))
            cbeam[0].set_3d_properties((stj[pos, 2], dtj[pos, 2]),'z')
            for i in range(4):
                beampf[i][0].set_data(*zip(stj[pos, :2],dverts[pos][0][i][:2]))
                beampf[i][0].set_3d_properties((stj[pos, 2], dverts[pos][0][i][2]), 'z')
            dcube.set_verts(dverts[pos])
            ang = geo.angles[pos] / np.pi * 180
            title.set_text(
                f"CBCT geometry in scale - Source at angle [{ang[0]:.1f}, {ang[1]:.1f}, {ang[2]:.1f}]°\n\n"
                f"Detector rotation [{roll[pos]:.1f}, {pitch[pos]:.1f}, {yaw[pos]:.1f}]°\n\n"
                f"Detector offset [{geo.offDetector[pos, 0]:.1f}, {geo.offDetector[pos, 1]:.1f}] mm")
        elif rotation == "obj":
            ang = -geo.angles[pos] / np.pi * 180
            ax.set_title(
                f"CBCT geometry in scale - Object at angle [{ang[0]:.1f}, {ang[1]:.1f}, {ang[2]:.1f}]°\n\n"
                f"Detector rotation [{roll[pos]:.1f}, {pitch[pos]:.1f}, {yaw[pos]:.1f}]°\n\n"
                f"Detector offset [{geo.offDetector[pos, 0]:.1f}, {geo.offDetector[pos, 1]:.1f}] mm")
    
    if animate:
        ani = animation.FuncAnimation(fig, update, len(angles), interval=100)
        if fname:
            fname += '_geometry'
            for writer in ['ffmpeg','pillow','imagemagick']:
                try:
                    ani.save(f'{fname}.mp4' if writer == 'ffmpeg' else f'{fname}.gif', writer=writer, fps=30)
                    break
                except ValueError:
                    continue
        return ani
    else:
        return ax


def calCube(centre, size, R='eye', offcent=None):
    """
    Calculate vertices of a cuboid centered at 'centre' with given 'size' and rotation 'R'.
    Args:
        centre: array-like, shape (3,) or (n,3) - center position(s)
        size: float - cuboid size
        R: str or array-like, shape (3,3) or (n,3,3) - rotation matrix, 'eye' for identity
        offcent: array-like, shape (3,) - offset from center, defaults to [0,0,0]
    Returns:
        list of vertex arrays representing cuboid faces
    """
    # Define cube corners once as a constant
    CORNERS = np.array([[-1, -1, -1],
                        [ 1, -1, -1],
                        [ 1,  1, -1],
                        [-1,  1, -1],
                        [-1, -1,  1],
                        [ 1, -1,  1],
                        [ 1,  1,  1],
                        [-1,  1,  1]], dtype=np.float64)
    
    # Define face indices once as a constant
    FACE_INDICES = np.array([[1, 2, 6, 5],  # front
                            [0, 1, 2, 3],   # bottom
                            [4, 5, 6, 7],   # top
                            [0, 1, 5, 4],   # left
                            [2, 3, 7, 6],   # right
                            [0, 3, 7, 4]])  # back
    
    # Convert inputs to numpy arrays and set defaults
    centre = np.asarray(centre, dtype=np.float64)
    offcent = np.zeros(3, dtype=np.float64) if offcent is None else np.asarray(offcent, dtype=np.float64)
    is_batched = centre.ndim > 1
    
    # Pre-compute scaled corners
    scaled_corners = CORNERS * (size / 2) + offcent
    
    # Compute vertices
    if is_batched:
        n = centre.shape[0]
        verts = []
        # Handle rotation
        if isinstance(R, str):
            rotation = np.eye(3)
            rotated = scaled_corners @ rotation.T
            for i in range(n):
                corners = rotated + centre[i]
                verts.append(corners[FACE_INDICES])
        else:
            R = np.asarray(R)
            for i in range(n):
                corners = scaled_corners @ R[i].T + centre[i]
                verts.append(corners[FACE_INDICES])
    else:
        # Handle single cuboid
        if isinstance(R, str):
            rotation = np.eye(3)
        else:
            rotation = np.asarray(R)
        corners = scaled_corners @ rotation.T + centre
        verts = corners[FACE_INDICES].tolist()
    return verts


# self testing 
if __name__ == "__main__":
    n_voxel_z = 128
    n_voxel_y = 256
    n_voxel_x = 512
    n_detector_u = 400
    n_detector_v = 300
    off_detector_u = 0  # 500
    off_detector_v = 0  # 500
    off_origin_x = 0  # 300
    off_origin_y = 0  # 100
    off_origin_z = 0  # 100
    rot_detector_roll = 0 # degrees
    rot_detector_pitch = 0
    rot_detector_yaw = 0

    geo = tigre.geometry(mode="cone", default=True)
    geo.nVoxel = np.array([n_voxel_z, n_voxel_y, n_voxel_x])
    geo.sVoxel = geo.nVoxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.nDetector = np.array([n_detector_v, n_detector_u])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.offDetector = np.array([off_detector_v, off_detector_u])
    geo.offOrigin = np.array([off_origin_z, off_origin_y, off_origin_x])
    geo.rotDetector = np.radians([rot_detector_roll, rot_detector_pitch, rot_detector_yaw]) 
    print(geo)
    angles = np.linspace(0.0, 2*np.pi, 100)
    pos = np.argmin(abs(angles < np.pi / 6))
    ani1=animate_geometry(geo, angles, pos, rotation="obj", animate=True, fname="Rotate object")
    ani1
    plt.show()
    ani2=animate_geometry(geo, angles, pos, rotation="SD", animate=True, fname="Rotate source and detector")
    ani2
    plt.show()
