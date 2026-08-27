"""Exact per-view TIGRE geometry for a TILTED ROTATION AXIS.

WHY THIS EXISTS. A tilted rotation axis cannot be expressed by any rigid
detector transform - not offsets, and not rotDetector. A utility that tried
(axis tilt as per-view offOrigin/offDetector) was deleted after its own
validation falsified it: the source traces a TILTED CIRCLE in the object
frame, which changes its radius, azimuth AND the detector's orientation, and
translations capture one term of three. The correct expression needs PER-VIEW
geometry, which TIGRE's kernels already consume (dRoll/dPitch/dYaw, DSO, DSD,
offDetector and offOrigin are all per-projection arrays).

THE MODEL. The stage rotates the OBJECT by angle theta about a unit axis
a = T @ z_hat, where T = Rx(tilt_x) @ Ry(tilt_y) is the tilt away from the
ideal vertical. Reconstruct in the AXIS-ALIGNED frame A (the rotation axis IS
z there) - the frame in which the volume has a clean vertical axis, which is
what every downstream consumer assumes. Seen from A, the lab-fixed source and
detector orbit the axis; with the EMPIRICALLY PINNED convention that TIGRE
rotates its source/detector by Rz(+angle):

    X_i = Rz(+theta_i) @ T.T @ X_lab      for X in {S, C, u_hat, v_hat}

(Conventions were pinned by single-ball probes against tigre.Ax, not read
from source: axis order (z, y, x); +y -> +u index; +z -> +v index; pixel k
centred at (k + 0.5 - N/2) * d; world rotation Rz(+theta). Reading the CUDA
gave the wrong sign twice before measuring settled it.)

Writing q = T.T @ x_hat, q_h = |q_xy|, the source rides a circle of radius
DSO*q_h at CONSTANT height DSO*q_z; the detector centre rides the opposite
circle of radius (DSD-DSO)*q_h at height -(DSD-DSO)*q_z; both centres stay on
a line through the axis. The detector's orientation PRECESSES per view.

MAPPING TO TIGRE, per view i (exact, no small-angle steps):

  angles_i      = theta_i + azimuth(q)     (constant phase)
  DSO_i         = DSO * q_h
  DSD_i         = DSD * q_h                (then the de-rotated centre lands
                                            exactly at -(DSD_i - DSO_i) in x)
  offOrigin_z   = -DSO * q_z               (TIGRE has no source height; shift
                                            the frame so the source is at z=0)
  offDetector_i = the de-rotated centre's residual (y, z) displacement
  rotDetector_i = Euler angles of the de-rotated detector orientation in the
                  kernel convention R = Rz(rot[2]) Ry(rot[1]) Rx(rot[0]),
                  solved from the OBSERVABLE axes (R@y_hat = u, R@z_hat = v)

LIMITS (v1, deliberate): COR and pre-existing offDetector must be zero -
composing a displaced axis with a tilted one is separate algebra, and
validating one thing at a time is the lesson this module's deleted
predecessor teaches. Cone mode only.

Validation: tests/test_tilted_axis_geometry.py compares tigre.Ax ball
centroids under the built geometry against an analytic pinhole projection of
the physical model. The predecessor scored 12.48 px on this style of test;
the acceptance bar here is sub-pixel.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _tilt_matrix(tilt_x, tilt_y):
    """T mapping the ideal vertical to the physical rotation axis."""
    return (Rotation.from_euler("x", tilt_x) * Rotation.from_euler("y", tilt_y)
            ).as_matrix()


def tilted_axis_geo(geo, angles, tilt_x, tilt_y):
    """Fill `geo` with per-view arrays for a tilted rotation axis.

    Returns (geo, angles_out): reconstruct with BOTH, e.g.
        geo, ang = tilted_axis_geo(geo, angles, tx, ty)
        rec = tigre.algorithms.fdk(proj, geo, ang)
    """
    angles = np.asarray(angles, dtype=np.float64).ravel()
    n = angles.size
    DSD = float(np.ravel(geo.DSD)[0])
    DSO = float(np.ravel(geo.DSO)[0])
    if getattr(geo, "mode", "cone") != "cone":
        raise ValueError("tilted_axis_geo: cone mode only")
    if np.any(np.ravel(getattr(geo, "COR", 0.0))) or \
       np.any(np.ravel(getattr(geo, "offDetector", 0.0))):
        raise NotImplementedError(
            "tilted_axis_geo v1: COR/offDetector must be zero - composing a "
            "displaced axis with a tilted one is not validated yet")

    T = _tilt_matrix(tilt_x, tilt_y)
    Tt = T.T
    q = Tt @ np.array([1.0, 0.0, 0.0])
    q_h = float(np.hypot(q[0], q[1]))
    phase = float(np.arctan2(q[1], q[0]))
    z_src = DSO * q[2]                       # constant source height

    geo.DSO = np.full(n, DSO * q_h, dtype=np.float64)
    geo.DSD = np.full(n, DSD * q_h, dtype=np.float64)

    # offOrigin component order is (z, y, x) - pinned empirically: putting
    # +10 in index 0 moves a central ball's image to exactly the v predicted
    # for a +z volume shift, index 1 moves u (y), index 2 is invisible at
    # azimuth 0 (x, along the beam). NOT the (x, y, z) one might assume.
    off_origin = np.zeros((n, 3), dtype=np.float64)
    off_origin[:, 0] = -z_src               # volume centre in the source-z=0 frame

    # Lab detector entities. u/v here are the OBSERVABLE directions of
    # increasing pixel index (pinned empirically: +y -> +u, +z -> +v).
    C_lab = np.array([-(DSD - DSO), 0.0, 0.0])
    u_lab = np.array([0.0, 1.0, 0.0])
    v_lab = np.array([0.0, 0.0, 1.0])

    rot_det = np.zeros((n, 3), dtype=np.float64)
    off_det = np.zeros((n, 2), dtype=np.float64)
    angles_out = (angles + phase)

    for i, th in enumerate(angles):
        Rw = Rotation.from_euler("z", th).as_matrix()          # world at view
        Rd = Rotation.from_euler("z", -(th + phase)).as_matrix()  # de-rotate
        C = Rd @ (Rw @ (Tt @ C_lab))
        u = Rd @ (Rw @ (Tt @ u_lab))
        v = Rd @ (Rw @ (Tt @ v_lab))

        # Orientation: solve R with R@y=u, R@z=v (observable axes), column
        # form R = [u x v | u | v] ... x_hat column = y_hat x z_hat image.
        M = np.column_stack([np.cross(u, v), u, v])
        e = Rotation.from_matrix(M).as_euler("ZYX")
        rot_det[i] = e[::-1]                # -> [rot0, rot1, rot2]

        # Centre residual after TIGRE places the nominal centre at
        # (-(DSD_i - DSO_i), 0, 0) in the source-z=0 frame.
        off_det[i, 1] = C[1]                # U on y
        off_det[i, 0] = C[2] - z_src        # V on z, frame shifted by z_src
        # C[0] + (DSD_i - DSO_i) vanishes identically by the DSD_i choice.

    geo.offOrigin = off_origin
    geo.offDetector = off_det
    geo.rotDetector = rot_det
    return geo, angles_out.astype(np.float32)


def project_points_tilted(points, geo_nominal, angles, tilt_x, tilt_y):
    """Analytic pinhole projection under the PHYSICAL tilted-axis model.

    Ground truth for the validation test - no TIGRE involved. `points` are in
    the axis-aligned frame; returns (n_views, n_points, 2) pixel (u, v)
    indices. Conventions pinned empirically against tigre.Ax at tilt zero:
    world rotation Rz(+theta), +y -> +u, +z -> +v, pixel k centred at
    (k + 0.5 - N/2) * d.
    """
    DSD = float(np.ravel(geo_nominal.DSD)[0])
    DSO = float(np.ravel(geo_nominal.DSO)[0])
    du = float(geo_nominal.dDetector[1])
    dv = float(geo_nominal.dDetector[0])
    nu = int(geo_nominal.nDetector[1])
    nv = int(geo_nominal.nDetector[0])

    T = _tilt_matrix(tilt_x, tilt_y)
    S_lab = np.array([DSO, 0.0, 0.0])
    C_lab = np.array([-(DSD - DSO), 0.0, 0.0])
    u_lab = np.array([0.0, 1.0, 0.0])
    v_lab = np.array([0.0, 0.0, 1.0])
    pts = np.asarray(points, dtype=np.float64)
    out = np.empty((len(angles), len(pts), 2))
    for i, th in enumerate(np.asarray(angles, dtype=np.float64)):
        Rw = Rotation.from_euler("z", th).as_matrix()
        S = Rw @ (T.T @ S_lab)
        C = Rw @ (T.T @ C_lab)
        u = Rw @ (T.T @ u_lab)
        v = Rw @ (T.T @ v_lab)
        nrm = np.cross(u, v)
        for j, p in enumerate(pts):
            ray = p - S
            t = np.dot(C - S, nrm) / np.dot(ray, nrm)
            hit = S + t * ray - C
            out[i, j, 0] = np.dot(hit, u) / du + nu / 2.0 - 0.5
            out[i, j, 1] = np.dot(hit, v) / dv + nv / 2.0 - 0.5
    return out
