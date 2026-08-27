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

Because Rz(theta) and the de-rotation Rz(-(theta + phase)) compose to the
CONSTANT Rz(-phase), every per-view quantity is in fact the same for all
views: a tilted axis is a constant re-expression of the rig (DSO, DSD,
offOrigin z, offDetector, rotDetector) plus a constant phase on the angles.
The arrays are still emitted per view because that is TIGRE's interface.

MAPPING TO TIGRE, per view i (exact, no small-angle steps):

  angles_i      = theta_i + azimuth(S)     (constant phase)
  DSO_i         = |S_xy|                   (source's horizontal radius)
  DSD_i         = S_x - C_x                (de-rotated; puts the nominal
                                            detector centre where TIGRE
                                            expects it, at -(DSD_i - DSO_i))
  offOrigin_z   = -S_z                     (TIGRE has no source height; shift
                                            the frame so the source is at z=0)
  offDetector_i = the de-rotated centre's residual (y, z) displacement
  rotDetector_i = Euler angles of the de-rotated detector orientation in the
                  kernel convention R = Rz(rot[2]) Ry(rot[1]) Rx(rot[0]),
                  solved from the OBSERVABLE axes (R@y_hat = u, R@z_hat = v)

DISPLACED AXIS (COR) AND DETECTOR OFFSET - composed since 2026-08-27. TIGRE's
COR is a rigid lateral shift of the source AND detector relative to the
rotation axis (the CUDA adds COR to both S and the detector point along the
in-plane direction perpendicular to the beam, Siddon_projection.cu and
voxel_backprojection.cu alike); offDetector shifts the detector alone. Both
are lab-fixed, so they enter here as the lab positions

    S_lab = (DSO, COR, 0)
    C_lab = (-(DSD - DSO), COR + offDetector_u, offDetector_v)

before the tilt mapping, and the output geometry carries COR = 0 with the
displacement folded into offDetector. At zero tilt this reduces EXACTLY to
TIGRE's documented equivalence offDetector_u + (DSD/DSO) * COR (to first
order in COR/DSO; the exact form is what the rotation produces), which is
what tests/test_tilted_axis_cor.py pins against tigre.Ax with a plain COR
geometry before any tilt enters. Cone mode only.

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


def _lab_entities(geo):
    """Lab-frame source, detector centre and detector axes, incl. COR and
    offDetector (TIGRE conventions: COR shifts source+detector along +y,
    offDetector = (v, u) shifts the detector alone)."""
    DSD = float(np.ravel(geo.DSD)[0])
    DSO = float(np.ravel(geo.DSO)[0])
    cor = float(np.ravel(getattr(geo, "COR", 0.0))[0]) if np.size(
        getattr(geo, "COR", 0.0)) else 0.0
    off = np.atleast_2d(np.asarray(getattr(geo, "offDetector", np.zeros(2)),
                                   dtype=np.float64))[0]
    off_v, off_u = float(off[0]), float(off[1])
    S_lab = np.array([DSO, cor, 0.0])
    C_lab = np.array([-(DSD - DSO), cor + off_u, off_v])
    u_lab = np.array([0.0, 1.0, 0.0])
    v_lab = np.array([0.0, 0.0, 1.0])
    return S_lab, C_lab, u_lab, v_lab


def tilted_axis_geo(geo, angles, tilt_x, tilt_y):
    """Fill `geo` with per-view arrays for a tilted rotation axis.

    `geo` may carry a scalar COR and a (v, u) offDetector; both are folded
    into the per-view geometry (COR is returned as zeros - do not add it
    again). Returns (geo, angles_out): reconstruct with BOTH, e.g.
        geo, ang = tilted_axis_geo(geo, angles, tx, ty)
        rec = tigre.algorithms.fdk(proj, geo, ang)
    """
    angles = np.asarray(angles, dtype=np.float64).ravel()
    n = angles.size
    if getattr(geo, "mode", "cone") != "cone":
        raise ValueError("tilted_axis_geo: cone mode only")
    if np.size(getattr(geo, "COR", 0.0)) > 1 and np.ptp(np.ravel(geo.COR)) != 0:
        raise NotImplementedError("tilted_axis_geo: per-view COR arrays are not supported")
    off = np.asarray(getattr(geo, "offDetector", np.zeros(2)), dtype=np.float64)
    if off.ndim > 1 and np.ptp(off, axis=0).any():
        raise NotImplementedError("tilted_axis_geo: per-view offDetector arrays are not supported")

    S_lab, C_lab, u_lab, v_lab = _lab_entities(geo)
    T = _tilt_matrix(tilt_x, tilt_y)
    Tt = T.T

    S0 = Tt @ S_lab
    phase = float(np.arctan2(S0[1], S0[0]))
    Rd = Rotation.from_euler("z", -phase).as_matrix()   # de-rotation, constant
    S = Rd @ S0                                          # (|S_xy|, 0, S_z)
    C = Rd @ (Tt @ C_lab)
    u = Rd @ (Tt @ u_lab)
    v = Rd @ (Tt @ v_lab)

    dso_i = float(S[0])
    dsd_i = float(S[0] - C[0])
    z_src = float(S[2])

    geo.DSO = np.full(n, dso_i, dtype=np.float64)
    geo.DSD = np.full(n, dsd_i, dtype=np.float64)

    # offOrigin component order is (z, y, x) - pinned empirically: putting
    # +10 in index 0 moves a central ball's image to exactly the v predicted
    # for a +z volume shift, index 1 moves u (y), index 2 is invisible at
    # azimuth 0 (x, along the beam). NOT the (x, y, z) one might assume.
    off_origin = np.zeros((n, 3), dtype=np.float64)
    off_origin[:, 0] = -z_src               # volume centre in the source-z=0 frame

    # Orientation: solve R with R@y=u, R@z=v (observable axes), column
    # form R = [u x v | u | v].
    M = np.column_stack([np.cross(u, v), u, v])
    e = Rotation.from_matrix(M).as_euler("ZYX")
    rot_det = np.tile(e[::-1], (n, 1))      # -> [rot0, rot1, rot2]

    # Centre residual after TIGRE places the nominal centre at
    # (-(DSD_i - DSO_i), 0, 0) in the source-z=0 frame. C[0] matches by the
    # DSD_i choice; the (y, z) remainder is the detector offset.
    off_det = np.zeros((n, 2), dtype=np.float64)
    off_det[:, 1] = C[1]                    # U on y
    off_det[:, 0] = C[2] - z_src            # V on z, frame shifted by z_src

    geo.offOrigin = off_origin
    geo.offDetector = off_det
    geo.rotDetector = rot_det
    geo.COR = np.zeros(n, dtype=np.float64)   # folded into offDetector above
    return geo, (angles + phase).astype(np.float32)


def project_points_tilted(points, geo_nominal, angles, tilt_x, tilt_y):
    """Analytic pinhole projection under the PHYSICAL tilted-axis model.

    Ground truth for the validation test - no TIGRE involved. `points` are in
    the axis-aligned frame; returns (n_views, n_points, 2) pixel (u, v)
    indices. Honours geo_nominal's scalar COR and offDetector the same way
    tilted_axis_geo does (lab-frame shifts of source/detector). Conventions
    pinned empirically against tigre.Ax at tilt zero: world rotation
    Rz(+theta), +y -> +u, +z -> +v, pixel k centred at (k + 0.5 - N/2) * d.
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
