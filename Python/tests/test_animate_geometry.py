
## test    

import numpy as np
from matplotlib import pyplot as plt
import tigre

# # plot 1, check zero angle position, scan rotation direction, detector offset

# geo = tigre.geometry(nVoxel=np.array([256,128,256]),default=True)
# geo.sVoxel = geo.nVoxel * np.array([1,1,1]) # (z,y,x)
# geo.nDetector = np.array([256,128])
# geo.dDetector = np.array([0.8, 0.8])*2               
# geo.sDetector = geo.dDetector * geo.nDetector 
# geo.offDetector=np.array([100,-100]) # viewing from S, D move (up, right) => (v, u)
# geo.rotDetector=np.array([30,0,0])/180*np.pi # [roll, pitch, yaw] viewing from S to D
# geo.offOrigin = np.array([0,0,0]) # (z,y,x)
# geo.COR=0
# angles=np.linspace(0,np.pi,100)
# ani1=tigre.animate_geometry(geo,angles,0,animate=True)  # angle=0, S is at (x=DSO, y=0, z=0)
# ani1
# # confirm the plot with projection
# from scipy.io import loadmat
# head=loadmat('../tigre/data/head.mat')['img'].transpose(2,1,0).copy()
# head=head[:,:128,:].copy()
# proj = tigre.Ax(head,geo,angles)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(head[:,:,128],origin='lower')
# plt.title('dim2=128')
# plt.ylabel('dim0 ->')
# plt.xlabel('dim1 ->')
# plt.subplot(1,2,2)
# plt.imshow(proj[0,:,:],origin='lower')
# plt.ylabel('v ->')
# plt.xlabel('u ->')


# # plot 2, check staticDetectorGeo() for tomosymthesis setup

# geo = tigre.geometry_default()
# geo.DSO = 400
# geo.DSD = 800
# geo.offDetector=np.array([20, 50])
# geo.rotDetector=np.radians( np.array([20, 0, 0]) ) # (roll, pitch, yaw)
# angles=np.radians( np.linspace(-60.0, 60.0, 31) )
# #geo.offOrigin = np.array([50, 30, 0]) # (z,y,x)

# #geo = tigre.staticDetectorGeo(geo, angles, 90)

# s_pos = np.column_stack( (geo.DSO * np.cos(angles), geo.DSO * np.sin(angles), 0*angles) )
# DS = geo.DSO - geo.DSD
# #d_pos = np.column_stack( (DS * np.cos(angles), DS * np.sin(angles), 0*angles) )
# d_pos = np.column_stack( (DS + 0*angles, 0 * angles, 0*angles) )
# geo = tigre.ArbitrarySourceDetMoveGeo(geo, s_pos, d_pos)

# ani2=tigre.animate_geometry(geo, angles, 15, animate=True)
# ani2

 

# # plot 2a, tomosymthesis setup, with larger radian, source rotation centre offset
# geo = tigre.geometry_default()
# geo.DSO = 500
# geo.DSD = 800

# angles=np.linspace(-47.5,47.5,31)/180*np.pi
# r=1500 # larger radian
# soff=[-500,-100] # soruce rotation centre offset
# #soff=[-400,100] # soruce rotation centre offset
# s_pos = np.column_stack((r*np.cos(angles)+soff[0], r*np.sin(angles)+soff[1], 0*angles))
# DS = geo.DSO - geo.DSD
# d_pos = np.column_stack( (DS + 0*angles, 0 * angles, 0*angles) )
# geo21 = tigre.ArbitrarySourceDetMoveGeo(geo,s_pos, d_pos)
# ani21=tigre.animate_geometry(geo21, geo21.angles, 0, animate=True, fname='Tomosynthesis2')
# ani21

 
# plot 3, fixed target object and detector positions and orientations, source moving linearly

geo = tigre.geometry_default()
geo.DSO = 183
geo.DSD = 442.5
geo.nDetector=np.array([32, 512])
geo.dDetector=np.array([1, 1])*0.80078125
geo.sDetector=geo.nDetector*geo.dDetector

#df = np.linspace(-510,510,64)  # source position on 
df = np.linspace(-254, 254, 128)

#source on x-axis (when df=0)
pos = np.concatenate([-df, -df, -df])
rot = np.concatenate([df*0, df*0-90, df*0-180])
s_rot = 0

# pos = df
# s_rot = 0
# rot = 0

geo.offDetector = np.array([50,30])
geo.rotDetector = np.radians( np.array([0,0,10]) )

geo3 = tigre.staticDetLinearSourceGeo(geo,pos,s_rot=s_rot,rot=rot)

#ani3=tigre.animate_geometry(geo3, geo3.angles, rotation='obj', animate=True)
ani3=tigre.animate_geometry(geo3, geo3.angles, rotation='SD', animate=True)
ani3


# ## plot 4, helical CT
# geo = tigre.geometry_default(high_resolution=False)

# angles = np.linspace(0, 2 * np.pi, 100)
# angles = np.hstack([angles, angles, angles])  # loop 3 times

# # This makes it helical, axis order (z,y,x) for python
# geo.offOrigin = np.zeros((angles.shape[0], 3))
# geo.offOrigin[:, 0] = np.linspace(
#     -1024 / 2 + 128, 1024 / 2 - 128, angles.shape[0])

# ani4 = tigre.animate_geometry(geo, angles, 0, animate=True, fname='Helical_CT')
# ani4


# ## plot 5, fixed source and detector positions and orientations, object moving linearly, mimicing a cargo scanner

# geo = tigre.geometry_default()
# geo.DSO = 4700
# geo.DSD = 6500

# geo.nDetector = np.array([1024, 10])
# geo.dDetector = np.array([5, 5])
# geo.sDetector = geo.nDetector * geo.dDetector

# # 20 foot sea container
# geo.nVoxel = np.array( [518, 1212, 488] )
# geo.dVoxel = np.array([5, 5, 5])
# geo.sVoxel = geo.nVoxel * geo.dVoxel

# # source position
# s_pos = np.zeros((100,3))
# alpha = np.radians(13)
# s_pos[:, 0] = geo.DSO * np.cos(alpha)
# s_pos[:, 2] = geo.DSO * np.sin(alpha)

# # container movement
# geo.offOrigin = np.zeros((100,3))
# geo.offOrigin[:,1] = np.linspace(-3100, 3100, 100)
# geo.offOrigin[:,0] = 100

# #geo.rotDetector = np.radians( np.array([0, -39, 0]) )
# d_pos = None
# d_rot = np.zeros((100,3))
# d_rot[:,1] = -39/180*np.pi

# geo5 = tigre.ArbitrarySourceDetMoveGeo(geo, s_pos, d_pos, d_rot)

# ani5=tigre.animate_geometry(geo5, geo5.angles, 0, animate=True, fname='Container Scanner')
# ani5


# ## plot 6, fixed target object and source and detector are at 170 degree, source moving in an spiral

# geo = tigre.geometry_default()
# geo.DSO = 750
# geo.DSD = 1200

# n = 200
# t = np.linspace(0,np.pi*6,n)
# R = geo.DSD-geo.DSO
# s_pos = np.zeros((n,3))
# d_pos = np.zeros((n,3))
# d_rot = np.zeros((n,3))

# # source positions
# s_pos[:,0] = geo.DSO*np.cos(t)
# s_pos[:,1] = geo.DSO*np.sin(t)
# s_pos[:,2] = (geo.DSO/50 + t*10)

# # detector centre positions
# d_pos[:,0] = -R*np.cos(t+np.pi*17/18)
# d_pos[:,1] = -R*np.sin(t+np.pi*17/18)
# d_pos[:,2] = (R/50 + t*10)

# # detector rotation (roll,pitch,yaw) away from its normal direction towards (0, 0, 0)
# # d_rot[:,0] = np.cos(t)
# # d_rot[:,1] = np.sin(t)
# #d_rot[:,0] = t/np.pi*30

# geo6 = tigre.ArbitrarySourceDetMoveGeo(geo,s_pos,d_pos,d_rot)

# ani6=tigre.animate_geometry(geo6, geo6.angles, 0, animate=True, fname='Spiral Source')
# ani6
