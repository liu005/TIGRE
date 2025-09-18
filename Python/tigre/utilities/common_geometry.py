import numpy as np
import copy
from scipy.spatial.transform import Rotation
from .geometry import Geometry

# Tomosymthesis
def staticDetectorGeo(geo, angles, rot=0) -> Geometry:
    """
    # angles: angle off the axis between the source and detector centre (on x-y plane)
    #         when angles=0, the source is perpendicular to the detector
    # rot: rotation of both source and detector around origin
    """
    ngeo = copy.deepcopy(geo)
    R = ngeo.DSD-ngeo.DSO
    rot = np.deg2rad(rot)
    
    angles = np.asarray(angles)
    zero_arr = np.zeros_like(angles)
    cos_ang, sin_ang = np.cos(angles), np.sin(angles)   
    offDet0, offDet1 = ngeo.offDetector
    
    # Compute new DSD, DSO
    ngeo.DSD = ngeo.DSO + R * cos_ang - sin_ang * offDet1 
    ngeo.DSO = zero_arr + ngeo.DSO

    # Update offDetector
    ngeo.offDetector = np.column_stack((offDet0 + zero_arr, 
                                R * sin_ang + cos_ang * offDet1))

    # Apply detector rotation
    s_ang = np.column_stack((zero_arr, zero_arr, -angles))
    if hasattr(ngeo, 'rotDetector'):
        detector_ang = Rotation.from_euler('XYZ', s_ang).apply(ngeo.rotDetector)
        ngeo.rotDetector = detector_ang + s_ang
    else:
        ngeo.rotDetector = s_ang
    
    # Update angles
    ngeo.angles = np.column_stack((angles + rot, zero_arr, zero_arr))
    
    return ngeo

# Linear source
def staticDetLinearSourceGeo(geo, s_pos, s_rot=0, rot=0) -> Geometry:
    """
    # s_pos: distance from centre along source scanning linear trajectory 
    #        when s_pos = 0, source is aligned to the origin and detector center on x-axis
    # s_rot: rotation angle between the source linear trajectory and detector (looking from top on x-y plane, anticlockwise)
    # rot: source and detector rotation angle around the origin (looking from top on x-y plane, anticlockwise) 
    """
    ngeo = copy.deepcopy(geo)    
    s_pos = np.asarray(s_pos)
    s_rot = np.deg2rad(np.asarray(s_rot))
    rot = np.deg2rad(np.asarray(rot))
                   
    ang  = np.arctan2(s_pos*np.cos(s_rot), ngeo.DSO + s_pos*np.sin(s_rot)) 
    zero_arr = np.zeros_like(ang)
    R = ngeo.DSD - ngeo.DSO
    cos_ang, sin_ang = np.cos(ang), np.sin(ang)
    offDet0, offDet1 = ngeo.offDetector
    
    # Compute new DSD, DSO
    ngeo.DSO = np.sqrt(ngeo.DSO**2 + s_pos**2) 
    ngeo.DSD = ngeo.DSO + R * cos_ang - sin_ang * offDet1 
    
    # Update offDetector
    ngeo.offDetector = np.column_stack((offDet0 + zero_arr, 
                                R * sin_ang + cos_ang * offDet1))
    
    # Apply detector rotation
    s_ang = np.column_stack((zero_arr, zero_arr, -ang))
    if hasattr(ngeo, 'rotDetector'):
        proj_rot = Rotation.from_euler('XYZ', s_ang).apply(ngeo.rotDetector)
        ngeo.rotDetector = proj_rot + s_ang
    else:
        ngeo.rotDetector = s_ang

    # Update angles
    ngeo.angles = np.column_stack((ang + rot, zero_arr, zero_arr))
    
    return ngeo
        

def ArbitrarySourceDetMoveGeo(geo, s_pos, d_pos=None, d_rot=None) -> Geometry:
    """
    # Source and Detector can move arbitrarily while the object is fixed
    #
    #   Parameters
    #   ----------
    #   geo:   standard cone beam geometry
    #   s_pos: nx3 array, source movement coordinates (x,y,z), in mm
    #   d_pos: nx3 array, detector centre movement coordinates (x,y,z), in mm 
    #           default d_pos = oposite s_pos, detector centre facing origin
    #   d_rot: nx3 array, detector rotation angles (roll, pitch, yaw) in degrees, 
    #           default - no rotation, detector facing origin
    #   Note:
    #          a point source rotation has no effect, ignored
    #
    #   Returns
    #   -------
    #   geometry with arbitrarily specified movements of source and detector
    #
    """
    ngeo = copy.deepcopy(geo)
    
    # Source position
    s_pos = np.asarray(s_pos, dtype=np.float64)
    
    if s_pos.ndim != 2 or s_pos.shape[1] != 3:
        raise ValueError("Input s_pos should be an n x 3 array")
        
    n = s_pos.shape[0]
    DS = ngeo.DSD-ngeo.DSO
    
    # source euler angles away from x-axis in "XYZ" order
    s_ang = compute_xyz_euler(s_pos)
    
    # Compute source rotation
    Rs = Rotation.from_euler('XYZ', s_ang).as_matrix()

    # Detector position
    if d_pos is None:
        d_pos = np.einsum('nij,j->ni', Rs, np.array([-DS, 0, 0]))
    
    if d_rot is None:
        d_rot = np.zeros((n,3))
            
    d_pos = np.asarray(d_pos, dtype=np.float64)
    
    if s_pos.shape != d_pos.shape or s_pos.shape != d_rot.shape:
        raise ValueError("Inputs dimensions do not match")        
    
    # source and detector vector lengths and directions
    rs = np.linalg.norm(s_pos, axis=1, keepdims=True)
    rd = np.linalg.norm(d_pos, axis=1, keepdims=True)
    
    ns = s_pos / rs
    nd = d_pos / rd
            
    # check angles between OS and OD
    sd_cross = np.cross(ns, nd)
    # note: A x B = ||A||.||B||sin(theta)
    sd_ang = np.arcsin(sd_cross)
    if np.any(sd_ang >= np.pi/2):  
        raise RuntimeError("Source and detector must always on opposite sides")        
    
    # Compute detector offset in xyz
    d_offset = np.array([0, ngeo.offDetector[1], ngeo.offDetector[0]])
    
    # find planes perpendicular to the beam, while the detector centre positions are on the planes, 
    # and return the intersection points of the beam with the planes
    Int, D = find_intersection_planes(d_pos + d_offset, -ns)
       
    # Compute new DSD, DSO
    ngeo.DSO = rs.flatten()  
    ngeo.DSD = ngeo.DSO + abs(D) 
    
    # Update offDetector
    shift = d_pos - Int
    dd = np.linalg.norm(shift, axis=1, keepdims=True) * np.sign(shift)
    ngeo.offDetector = np.column_stack((dd[:,2], dd[:,1])) + Rotation.from_euler('XYZ', sd_ang).apply(d_offset)[:,2:0:-1]
        
    # Update rotDetector
    if hasattr(ngeo, 'rotDetector'):
        d_rot += ngeo.rotDetector
    else:
        # need this line to add attribute
        setattr(ngeo, 'rotDetector', None)
    ngeo.rotDetector = Rotation.from_euler('XYZ', -sd_ang).apply( d_rot - sd_ang )
        
    ngeo.angles = s_ang
    return ngeo


def find_intersection_planes(plane_points, normal, line_points=None):
    """
    Finds the equations of planes perpendicular to normal vectors that pass through a series given points.
    Then returns the intersections of the lines with direction normal to that planes

    Parameters:
    points : array-like, shape (3,n) - A point on the plane (x0, y0, z0)
    normal : array-like, shape (3,n) - The normal vector (A, B, C)

    Returns:
    intersection : The intersection of the plane with the beam
    D:    The plane equation coefficients (A, B, C, D) representing Ax + By + Cz = D
    """
    plane_points = np.asarray(plane_points)
    normal = np.asarray(normal)
    line_points = 0*plane_points if line_points == None else np.asarray(line_points)

    # Compute D = A*x0 + B*y0 + C*z0
    A, B, C = normal[:,0], normal[:,1], normal[:,2]
    D = np.einsum('ij,ij->i', normal, plane_points)

    # Compute denominator
    x0, y0, z0 = line_points[:,0], line_points[:,1], line_points[:,2]
    denominator = A**2 + B**2 + C**2
    
    # Compute t
    t = (D - (A*x0 + B*y0 + C*z0)) / denominator
    
    # Inersection points
    intersection = np.column_stack((x0 + A*t, y0 + B*t, z0 +C*t))

    return intersection, D


def compute_xyz_euler(points):
    """
    Converts 3D points (x, y, z) into Euler angles (XYZ convention) relative to (1,0,0).
    
    Parameters:
    points : array-like, shape (N,3) - List of 3D points [(x1, y1, z1), (x2, y2, z2), ...]
    
    Returns:
        euler_angles : ndarray, shape (N,3) - Euler angles (roll, pitch, yaw) for each point
        roll (phi)   -> Rotation about X-axis (set to 0, since it’s a direction vector)
        pitch (theta) -> Rotation about Y-axis
        yaw (psi)    -> Rotation about Z-axis
    """
    points = np.asarray(points)

    # Compute vector magnitudes
    r = np.linalg.norm(points[:,:2], axis=1)
    
    yaw = np.arctan2(points[:,1], points[:,0])  # Rotation about Z-axis
    pitch = np.arctan2(points[:,2], r)          # Rotation about Y-axis
    roll = np.zeros_like(yaw)                   # Roll set to zero since we're only aligning a direction vector

    return np.column_stack((roll, pitch, yaw))


def centre_of_rotation_3D(P1, D1, P2, D2):
    """
    Finds the centre of rotation between two 3D lines.
    
    P1, D1 : Point and direction vector of first line.
    P2, D2 : Point and direction vector of second line.
    
    Returns:
    - Centre of rotation if the lines are not parallel.
    - None if the lines are parallel (no unique centre).
    """
    # Convert to numpy arrays
    P1, D1, P2, D2 = map(np.array, (P1, D1, P2, D2))

    # Cross product to find normal vector
    N = np.cross(D1, D2)
    norm_N = np.linalg.norm(N)
    
    # If lines are parallel, no unique center of rotation
    if norm_N < 1e-6:
        return None

    # Compute the perpendicular vector between the lines
    V = P2 - P1
    
    # Solve for λ and μ (projection scalars)
    n = D1.shape[0]
    centre = np.zeros((n,3))
    for i in range(n):
        A = np.array([D1[i,:], -D2, N[i,:]]).T  # System matrix
        b = V[i,:]
    
        # Solve least squares if no exact solution
        try:
            lambda_mu_nu = np.linalg.lstsq(A, b, rcond=None)[0]
            λ, μ, _ = lambda_mu_nu
        except np.linalg.LinAlgError:
            return None
    
        # Compute closest points on both lines
        C1 = P1[i,:] + λ * D1[i,:]
        C2 = P2 + μ * D2

        # Centre of rotation (midpoint of closest points)
        centre[i] = (C1 + C2) / 2

    return centre


def ArbitrarySourceDetectorFixedObject(
        geometry: Geometry,
        focal_spot_position_mm: np.ndarray, 
        detector_center_position_mm: np.ndarray, 
        detector_line_direction: np.ndarray, 
        detector_column_direction: np.ndarray,
        origin_mm: np.ndarray = None,
        use_center_correction: bool = True) -> Geometry:
    """
    geo: Geometry object
    focal_spot_position_mm: position of the source, 
    detector_center_position_mm: position of the detector center, 
    detector_line_direction: detector line vector from pixel (0, 0) -> (0, 1), 
    detector_column_direction: detector column vector from pixel (0, 0) -> (1, 0),
    origin_mm: origin of the ct trajectory. The source and detector positions are translated with this value. Defaults to: None.
    use_center_correction: Calculate an arbitrary origin of the trajectory. Defaults to: True.
    """

    # Assumption: CT trajectory has one rotation center.
    number_of_projection = focal_spot_position_mm.shape[0]
    
    if origin_mm is None:
        origin_mm = np.zeros((3,))

    focal_spot_position_mm = focal_spot_position_mm - origin_mm
    detector_center_position_mm = detector_center_position_mm - origin_mm
    
    if use_center_correction:
        trajectory_center_mm = calculate_trajectory_center_mm(
            focal_spot_position_mm, detector_center_position_mm)
        focal_spot_position_mm = focal_spot_position_mm - trajectory_center_mm
        detector_center_position_mm = detector_center_position_mm - trajectory_center_mm
    else:
        trajectory_center_mm = np.zeros((3,))

    if not use_center_correction:
        geometry.offOrigin = trajectory_center_mm.reshape((3, ))

    # source and detector are orthogonal. The angle is rotates the source from the x axis.

    # 1. find nearest point from source detector line to trajectory center
    source_detector_vector = detector_center_position_mm - focal_spot_position_mm
    fdd_mm = np.linalg.norm(source_detector_vector, axis=1).reshape((-1, 1))
    fod_mm = np.zeros_like(fdd_mm)
    source_detector_direction = source_detector_vector / fdd_mm
    
    nearest_point_mm = np.zeros_like(focal_spot_position_mm)
    detector_offsets_mm = np.zeros((number_of_projection, 2))
    
    euler_zyz = np.zeros_like(focal_spot_position_mm)
    euler_xyz = np.zeros_like(focal_spot_position_mm)
    first_rot = np.eye(3)

    for i in range(number_of_projection):
        nearest_point_mm[i] = perpendicular_point_on_line(
            trajectory_center_mm, focal_spot_position_mm[i], -source_detector_direction[i])
        fod_mm[i] = np.linalg.norm(nearest_point_mm[i] - focal_spot_position_mm[i])

        # 2. calculate the angle rotation + offset and check it!
        rotation_matrix = rotation_from_vecs(np.array([1, 0, 0]),-nearest_point_mm[i] +focal_spot_position_mm[i])
        if i == 0:
            first_rot = rotation_matrix.T
            first_rot = np.eye(3)
        angle = rotation_matrix @ first_rot
        if not np.isclose(np.linalg.det(rotation_matrix), 1, 0.01):
            raise ValueError('Rotation matrix must be right handed!')
        
        rotation_inverse = Rotation.from_matrix(rotation_matrix.T)

        offsets = rotation_inverse.apply(nearest_point_mm[i])
        fod_mm[i] += offsets[0]
        detector_offsets_mm[i, 0] = offsets[2] * 2
        detector_offsets_mm[i, 1] = offsets[1] * 2

        # calculate the relative rotation of the detector
        detector_matrix = np.eye(3)
        detector_matrix[:, 1] = detector_line_direction[i]
        detector_matrix[:, 2] = detector_column_direction[i]
        detector_matrix[:, 0] = np.cross(detector_matrix[:, 1], detector_matrix[:, 2])

        if not np.isclose(np.linalg.det(detector_matrix), 1, 0.01):
            raise ValueError('Rotation matrix must be right handed!')
        
        # the rotation is from the angle rotation -> "real" rotation
        relative_rotation_matrix = rotation_matrix.T @ detector_matrix
        # relative_rotation_matrix = detector_matrix @ rotation_matrix.T -> wrong
        relative_rotation = Rotation.from_matrix(relative_rotation_matrix)

        euler_zyz[i] = Rotation.from_matrix(angle).as_euler('zyz', False)
        euler_xyz[i] = relative_rotation.as_euler('xyz', False) 
    
    geometry.DSO = fod_mm
    geometry.DSD = fdd_mm
    geometry.offDetector = detector_offsets_mm
    geometry.rotDetector = euler_xyz
    geometry.angles = euler_zyz
    return geometry

    
# def euler_from_vecs(a_vec,b_vec,order="xyz"):
#     """
#     Calculate Euler angles from two vectors

#     Parameters
#     ----------
#     a_vec : np.array of (n,3) or (3,)
#         A vector(s).
#     b_vec : np.array of (n,3) or (3,)
#         B vector(s).
#     order : string, optional
#         Order of the euler angles in rotations. The default is "xyz".

#     Returns
#     -------
#     euler : np.array of (n,3) or (3,)
#         Euler angles of rotations.

#     """
#     na = 1 if a_vec.ndim<2 else a_vec.shape[0]
#     nb = 1 if b_vec.ndim<2 else b_vec.shape[0]
#     n = max(na,nb)
#     a_vec = np.repeat([a_vec], n, axis=0) if a_vec.ndim<2 else a_vec
#     b_vec = np.repeat([b_vec], n, axis=0) if b_vec.ndim<2 else b_vec
#     euler = np.zeros_like(a_vec,dtype=np.float64)
#     for i in range(n):
#         R = rotation_from_vecs(a_vec[i,:],b_vec[i,:])
#         euler[i,:] = Rotation.from_matrix(R).as_euler(order)
#         for j in range(3):
#             if abs(euler[i,j]) < 2e-8: # very small angle
#                 euler[i,j] = 0 
#     return euler


def rotation_from_vecs(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """
    # unit vectors
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    # dimension of the space and identity
    I = np.identity(u.size)
    # the cos angle between the vectors (dot product)
    c = np.dot(u, Ru)
    # a small number
    eps = 1.0e-9
    if np.abs(c - 1.0) < eps:
        # same direction
        return I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        v = np.cross(u, Ru)
        K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)    


def perpendicular_point_on_line(point: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray):
    ap = point - line_point
    dot_product = np.dot(ap, line_direction)
    line_squared = np.dot(line_direction, line_direction)
    return line_point + (dot_product / line_squared) * line_direction

def calculate_trajectory_center_mm(focal_spot_position_mm: np.ndarray, detector_center_position_mm: np.ndarray):
    number_of_projection = focal_spot_position_mm.shape[0]
    A = np.zeros((number_of_projection * 6, 5))
    b = np.zeros((number_of_projection * 6, 1))

    print(f'Calculated FOD / ODD')

    for i in range(0, number_of_projection * 6-1, 6):
        ii = i // 6
        source = focal_spot_position_mm[ii]
        detector = detector_center_position_mm[ii]
        direction = source - detector
        direction = direction / np.linalg.norm(direction)

        A[i, :] = np.array([1, 0, 0, direction[0], 0])
        b[i] = source[0]
        i=i+1

        A[i, :] = np.array([0, 1, 0, direction[1], 0])
        b[i] = source[1]
        i=i+1

        A[i, :] = np.array([0, 0, 1, direction[2], 0])
        b[i] = source[2]
        i=i+1

        A[i, :] = np.array([1, 0, 0, 0, -direction[0]])
        b[i] = detector[0]
        i=i+1

        A[i, :] = np.array([0, 1, 0, 0, -direction[1]])
        b[i] = detector[1]
        i=i+1

        A[i, :] = np.array([0, 0, 1, 0, -direction[2]])
        b[i] = detector[2]
        i=i+1            
    
    res = np.linalg.lstsq(A, b)
    trajectory_center_mm = np.array([res[0][0], res[0][1], res[0][2]]).reshape((1, 3))
    return trajectory_center_mm

