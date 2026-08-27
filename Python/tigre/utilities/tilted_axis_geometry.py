"""Exact per-view TIGRE geometry for a TILTED ROTATION AXIS.

WHY THIS EXISTS. A tilted rotation axis cannot be expressed by any rigid
detector transform - not offsets, and not rotDetector. A utility that tried
(axis tilt as per-view offOrigin/offDetector) was deleted after its own
validation falsified it: the source traces a TILTED CIRCLE in the object
frame, which changes its radius, azimuth AND the detector's orientation, and
translations capture one term of three. The correct expression needs PER-VIEW
geometry, which TIGRE's kernels already consume (dRoll/dPitch/dYaw, DSO, DSD,
COR, offDetector and offOrigin are all per-projection arrays).

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

Because Rz(theta) and the de-rotation Rz(-(theta + phase)) compose to the
CONSTANT Rz(-phase), every per-view quantity is in fact the same for all
views: a tilted axis is a constant re-expression of the rig (DSO, DSD, COR,
offOrigin z, offDetector, rotDetector) plus a constant phase on the angles.
The arrays are still emitted per view because that is TIGRE's interface.

MAPPING TO TIGRE (exact, no small-angle steps). De-rotate by the azimuth of the
UNDISPLACED source direction T.T @ x_hat, so that TIGRE's own semantics can be
read off directly - TIGRE puts the source at (DSO, COR, 0) and the detector
centre at (-(DSD-DSO), COR + offDetector_u, offDetector_v):

  angles_i      = theta_i + phase
  DSO_i         = S_x            COR_i = S_y
  DSD_i         = S_x - C_x      offDetector_u = C_y - COR_i
  offOrigin_z   = -S_z           offDetector_v = C_z - S_z
  rotDetector_i = Euler angles of the de-rotated detector orientation in the
                  kernel convention R = Rz(rot[2]) Ry(rot[1]) Rx(rot[0]),
                  solved from the OBSERVABLE axes (R@y_hat = u, R@z_hat = v)

At ZERO TILT this returns the input geometry EXACTLY - COR, offDetector and
rotDetector included - which matters beyond tidiness: TIGRE's Wang
(offset-detector) weighting is gated on offDetector_u == 0, so a builder that
folded COR into offDetector_u switched Wang ON for a nominally centred
detector and halved the reconstructed intensity (each opposing-ray pair
counted once against the FDK normalisation's twice). Measured 2026-08-27 on
scan 18 (mean 0.0126 -> 0.0063) before this was changed.

DISPLACED AXIS (COR), DETECTOR OFFSET AND DETECTOR ROTATION - composed. COR is
a rigid lateral shift of source AND detector (Siddon_projection.cu and
voxel_backprojection.cu add it to both), offDetector shifts the detector alone,
rotDetector rotates the detector axes about the detector centre with
R = Rz(rot[2]) Ry(rot[1]) Rx(rot[0]) on (beam, u, v) (bindings: dYaw <- rot[0]
is the in-plane roll about the beam). All three are lab-fixed, so they enter as
the lab entities before the tilt mapping:

    S_lab = (DSO, COR, 0)
    C_lab = (-(DSD - DSO), COR + offDetector_u, offDetector_v)
    u_lab = R @ y_hat,  v_lab = R @ z_hat

tests/test_tilted_axis_cor.py pins each against tigre.Ax at zero tilt before
any tilt test runs. Cone mode only.

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


def _scalar0(v, default=0.0):
    a = np.ravel(np.asarray(v if v is not None else default, dtype=np.float64))
    return float(a[0]) if a.size else float(default)


def _lab_entities(geo):
    """Lab-frame source, detector centre and detector axes, incl. COR,
    offDetector and rotDetector (TIGRE conventions, see module docstring)."""
    DSD = _scalar0(geo.DSD)
    DSO = _scalar0(geo.DSO)
    cor = _scalar0(getattr(geo, "COR", 0.0))
    off = np.atleast_2d(np.asarray(getattr(geo, "offDetector", np.zeros(2)),
                                   dtype=np.float64))[0]
    off_v, off_u = float(off[0]), float(off[1])
    rot = np.atleast_2d(np.asarray(getattr(geo, "rotDetector", np.zeros(3)),
                                   dtype=np.float64))[0]
    R = Rotation.from_euler("ZYX", rot[[2, 1, 0]]).as_matrix()
    S_lab = np.array([DSO, cor, 0.0])
    C_lab = np.array([-(DSD - DSO), cor + off_u, off_v])
    u_lab = R @ np.array([0.0, 1.0, 0.0])
    v_lab = R @ np.array([0.0, 0.0, 1.0])
    return S_lab, C_lab, u_lab, v_lab


def _reject_varying(name, arr, row_ndim):
    """Refuse per-view arrays whose rows differ: the tilt composes ONE rig.
    `row_ndim` is the dimensionality of a single view's value (0 for
    COR/DSD/DSO, 1 for the (v, u) offDetector row and the rotDetector triple)
    so a plain one-row value is never mistaken for varying views."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim > row_ndim and a.shape[0] > 1 and np.ptp(a, axis=0).any():
        raise NotImplementedError(
            "tilted_axis_geo: per-view %s arrays that vary across views are "
            "not supported (the tilt is composed with ONE rig)" % name)


def tilted_axis_geo(geo, angles, tilt_x, tilt_y):
    """Fill `geo` with per-view arrays for a tilted rotation axis.

    `geo` may carry COR, offDetector and rotDetector (scalars / one row or
    constant per-view arrays); all are composed. Returns (geo, angles_out):
    reconstruct with BOTH, e.g.
        geo, ang = tilted_axis_geo(geo, angles, tx, ty)
        rec = tigre.algorithms.fdk(proj, geo, ang)
    """
    angles = np.asarray(angles, dtype=np.float64).ravel()
    n = angles.size
    if getattr(geo, "mode", "cone") != "cone":
        raise ValueError("tilted_axis_geo: cone mode only")
    for name, row_ndim in (("COR", 0), ("DSD", 0), ("DSO", 0),
                           ("offDetector", 1), ("rotDetector", 1)):
        if hasattr(geo, name):
            _reject_varying(name, getattr(geo, name), row_ndim)

    S_lab, C_lab, u_lab, v_lab = _lab_entities(geo)
    Tt = _tilt_matrix(tilt_x, tilt_y).T

    # De-rotate by the azimuth of the UNDISPLACED source direction, so the
    # residual lateral displacement reads off as TIGRE's COR (see docstring).
    q = Tt @ np.array([1.0, 0.0, 0.0])
    phase = float(np.arctan2(q[1], q[0]))
    Rd = Rotation.from_euler("z", -phase).as_matrix()   # constant de-rotation
    S = Rd @ (Tt @ S_lab)
    C = Rd @ (Tt @ C_lab)
    u = Rd @ (Tt @ u_lab)
    v = Rd @ (Tt @ v_lab)

    dso_i, cor_i, z_src = float(S[0]), float(S[1]), float(S[2])
    dsd_i = float(S[0] - C[0])

    geo.DSO = np.full(n, dso_i, dtype=np.float64)
    geo.DSD = np.full(n, dsd_i, dtype=np.float64)
    geo.COR = np.full(n, cor_i, dtype=np.float64)

    # offOrigin component order is (z, y, x) - pinned empirically: putting
    # +10 in index 0 moves a central ball's image to exactly the v predicted
    # for a +z volume shift, index 1 moves u (y), index 2 is invisible at
    # azimuth 0 (x, along the beam). NOT the (x, y, z) one might assume.
    off_origin = np.zeros((n, 3), dtype=np.float64)
    off_origin[:, 0] = -z_src               # volume centre in the source-z=0 frame

    # Orientation: solve R with R@y=u, R@z=v (observable axes), column
    # form R = [u x v | u | v]; TIGRE order [rot0, rot1, rot2] = reversed ZYX.
    M = np.column_stack([np.cross(u, v), u, v])
    e = Rotation.from_matrix(M).as_euler("ZYX")
    rot_det = np.tile(e[::-1], (n, 1))

    # Detector centre residuals against TIGRE's placement at
    # (-(DSD_i - DSO_i), COR_i + offU, offV) in the source-z=0 frame.
    off_det = np.zeros((n, 2), dtype=np.float64)
    off_det[:, 1] = C[1] - cor_i            # U on y
    off_det[:, 0] = C[2] - z_src            # V on z, frame shifted by z_src

    geo.offOrigin = off_origin
    geo.offDetector = off_det
    geo.rotDetector = rot_det
    return geo, (angles + phase).astype(np.float32)


def project_points_tilted(points, geo_nominal, angles, tilt_x, tilt_y):
    """Analytic pinhole projection under the PHYSICAL tilted-axis model.

    Ground truth for the validation test - no TIGRE involved. `points` are in
    the axis-aligned frame; returns (n_views, n_points, 2) pixel (u, v)
    indices. Honours geo_nominal's COR, offDetector and rotDetector the same
    way tilted_axis_geo does (lab-frame shifts/rotation of source/detector).
    Conventions pinned empirically against tigre.Ax at tilt zero: world
    rotation Rz(+theta), +y -> +u, +z -> +v, pixel k centred at
    (k + 0.5 - N/2) * d.
    """
    du = float(geo_nominal.dDetector[1])
    dv = float(geo_nominal.dDetector[0])
    nu = int(geo_nominal.nDetector[1])
    nv = int(geo_nominal.nDetector[0])

    T = _tilt_matrix(tilt_x, tilt_y)
    S_lab, C_lab, u_lab, v_lab = _lab_entities(geo_nominal)
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
