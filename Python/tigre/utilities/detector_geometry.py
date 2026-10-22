"""Data models for nonuniform, modular, and curved detector surfaces.

Coordinates use TIGRE's lab frame and millimetres. Source positions are
per-view; detector pixel centers are per-view and indexed by (row, column).
"""

import numpy as np


def _source_positions(source_positions_mm):
    source = np.asarray(source_positions_mm, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or source.shape[0] == 0:
        raise ValueError("source_positions_mm must have shape (n_views, 3), n_views > 0")
    if not np.isfinite(source).all():
        raise ValueError("source_positions_mm must contain only finite values")
    source = np.ascontiguousarray(source, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("source_positions_mm is out of float32 range")
    return source


def _pixel_centers(pixel_centers_mm, n_views):
    centers = np.asarray(pixel_centers_mm, dtype=np.float64)
    if centers.ndim == 3 and centers.shape[-1] == 3:
        centers = np.broadcast_to(centers, (n_views,) + centers.shape)
    if (centers.ndim != 4 or centers.shape[0] != n_views
            or centers.shape[-1] != 3 or min(centers.shape[1:3]) <= 0):
        raise ValueError(
            "pixel_centers_mm must have shape (rows, columns, 3) or "
            "(n_views, rows, columns, 3)")
    if not np.isfinite(centers).all():
        raise ValueError("pixel_centers_mm must contain only finite values")
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    if not np.isfinite(centers).all():
        raise ValueError("pixel_centers_mm is out of float32 range")
    return centers


def _pixel_mask(mask, shape):
    n_views, n_rows, n_columns = shape
    if mask is None:
        return np.ones(shape, dtype=np.bool_)
    values = np.asarray(mask)
    if values.dtype.kind not in "biuf" or np.any((values != 0) & (values != 1)):
        raise ValueError("valid_mask must contain only boolean or 0/1 values")
    valid = values.astype(np.bool_, copy=False)
    if valid.shape == (n_rows, n_columns):
        valid = np.broadcast_to(valid, shape)
    if valid.shape != shape:
        raise ValueError(
            f"valid_mask must have shape ({n_rows}, {n_columns}) or "
            f"({n_views}, {n_rows}, {n_columns}); got {valid.shape}")
    return np.ascontiguousarray(valid)


class DetectorModule:
    """A detector module with local pixel centers and per-view rigid poses.

    ``pixel_centers_mm`` is shaped ``(rows, columns, 3)`` and may describe
    either a flat or curved panel, including nonuniform pixel spacing.
    ``pose_matrices`` maps local homogeneous coordinates to the lab frame and
    may be one ``(4, 4)`` transform or one transform per view. Integer offsets
    place the module in the common dense detector-array layout; unoccupied
    cells become invalid gap samples.
    """

    def __init__(self, pixel_centers_mm, pose_matrices, row_offset=0, column_offset=0):
        centers = np.asarray(pixel_centers_mm, dtype=np.float64)
        if (centers.ndim != 3 or centers.shape[-1] != 3
                or min(centers.shape[:2]) <= 0):
            raise ValueError("module pixel_centers_mm must have shape (rows, columns, 3)")
        if not np.isfinite(centers).all():
            raise ValueError("module pixel_centers_mm must contain only finite values")
        if not isinstance(row_offset, (int, np.integer)) or row_offset < 0:
            raise ValueError("row_offset must be a non-negative integer")
        if not isinstance(column_offset, (int, np.integer)) or column_offset < 0:
            raise ValueError("column_offset must be a non-negative integer")

        transforms = np.asarray(pose_matrices, dtype=np.float64)
        if transforms.shape == (4, 4):
            transforms = transforms[None, ...]
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError("pose_matrices must have shape (4, 4) or (n_views, 4, 4)")
        if not np.isfinite(transforms).all():
            raise ValueError("pose_matrices must contain only finite values")
        if not np.allclose(transforms[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("pose_matrices must be homogeneous rigid transforms")
        rotations = transforms[:, :3, :3]
        identity = np.eye(3)
        if (not np.allclose(rotations @ np.swapaxes(rotations, 1, 2), identity, atol=1e-5)
                or np.any(np.linalg.det(rotations) <= 0)):
            raise ValueError("pose_matrices must contain right-handed rigid rotations")

        self.pixel_centers_mm = np.ascontiguousarray(centers, dtype=np.float32)
        self.pose_matrices = np.ascontiguousarray(transforms, dtype=np.float32)
        self.row_offset = int(row_offset)
        self.column_offset = int(column_offset)

    def world_pixel_centers(self, n_views):
        if self.pose_matrices.shape[0] not in (1, n_views):
            raise ValueError(
                "module pose count must be one or match the source view count")
        poses = np.broadcast_to(self.pose_matrices, (n_views, 4, 4))
        local = np.concatenate(
            (self.pixel_centers_mm,
             np.ones(self.pixel_centers_mm.shape[:2] + (1,), dtype=np.float32)),
            axis=2)
        return np.einsum("vij,hwj->vhwi", poses, local)[..., :3]


class DetectorSurfaceGeometry:
    """Per-view source and detector pixel geometry in TIGRE's lab frame.

    ``pixel_centers_mm`` has shape ``(views, rows, columns, 3)`` and records
    the physical center of each measured pixel. A false ``valid_mask`` entry
    marks an absent sample such as a module gap. Geometry arrays are kept
    separate from projection values; callers can use :meth:`pack_module_data`
    to place module measurements into the corresponding masked dense layout.
    """

    def __init__(self, source_positions_mm, pixel_centers_mm, valid_mask=None,
                 module_slices=None):
        self.source_positions_mm = _source_positions(source_positions_mm)
        self.pixel_centers_mm = _pixel_centers(
            pixel_centers_mm, len(self.source_positions_mm))
        self.valid_mask = _pixel_mask(valid_mask, self.pixel_centers_mm.shape[:3])
        self._module_slices = None
        if module_slices is not None:
            n_rows, n_columns = self.valid_mask.shape[1:]
            checked = []
            for rows, columns in module_slices:
                rows, columns = slice(*rows) if not isinstance(rows, slice) else rows, \
                    slice(*columns) if not isinstance(columns, slice) else columns
                if not (0 <= rows.start < rows.stop <= n_rows
                        and 0 <= columns.start < columns.stop <= n_columns):
                    raise ValueError("module_slices must lie inside the detector array")
                checked.append((rows, columns))
            self._module_slices = tuple(checked)

    def subset(self, view_indices):
        """Return a surface restricted to the given views (data is copied)."""
        idx = np.asarray(view_indices, dtype=np.intp)
        out = object.__new__(DetectorSurfaceGeometry)
        out.source_positions_mm = np.ascontiguousarray(self.source_positions_mm[idx])
        out.pixel_centers_mm = np.ascontiguousarray(self.pixel_centers_mm[idx])
        out.valid_mask = np.ascontiguousarray(self.valid_mask[idx])
        out._module_slices = self._module_slices
        return out

    def mask_outside_volume(self, geo, margin_mm=0.0):
        """Return a surface whose valid mask also excludes pixels that see no volume.

        A pixel is dropped when the segment from the source to the pixel does not
        intersect the reconstruction box of ``geo`` (grown by ``margin_mm``), e.g.
        pixels outside the beam or past the volume edge. Existing masking is kept.
        """
        voxel = np.asarray(geo.dVoxel, dtype=np.float64)[::-1]
        count = np.asarray(geo.nVoxel, dtype=np.float64)[::-1]
        offset = np.asarray(geo.offOrigin, dtype=np.float64)
        if offset.ndim == 2:
            offset = offset[0]
        offset = offset[::-1]
        half = 0.5 * count * voxel + margin_mm
        lo, hi = offset - half, offset + half

        src = self.source_positions_mm.astype(np.float64)
        n_views = src.shape[0]
        hits = np.empty(self.valid_mask.shape, dtype=np.bool_)
        step = max(1, int(4e6 // max(1, self.valid_mask[0].size)))
        for start in range(0, n_views, step):
            stop = min(n_views, start + step)
            s = src[start:stop, None, None, :]
            p = self.pixel_centers_mm[start:stop].astype(np.float64)
            d = p - s
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - s) / d, (hi - s) / d
            inside = (s >= lo) & (s <= hi)
            parallel = d == 0
            t_near = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
            t_far = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
            enter = np.maximum(t_near.max(axis=-1), 0.0)
            leave = np.minimum(t_far.min(axis=-1), 1.0)
            hits[start:stop] = enter < leave
        out = self.subset(np.arange(n_views))
        out.valid_mask = np.ascontiguousarray(self.valid_mask & hits)
        return out

    @classmethod
    def from_modules(cls, source_positions_mm, modules):
        """Combine modules into one dense array, leaving gaps explicitly masked."""
        source = _source_positions(source_positions_mm)
        modules = tuple(modules)
        if not modules or not all(isinstance(m, DetectorModule) for m in modules):
            raise ValueError("modules must be a non-empty sequence of DetectorModule objects")

        placements = []
        n_rows = 0
        n_columns = 0
        for module in modules:
            centers = module.world_pixel_centers(len(source))
            row_end = module.row_offset + centers.shape[1]
            column_end = module.column_offset + centers.shape[2]
            n_rows = max(n_rows, row_end)
            n_columns = max(n_columns, column_end)
            placements.append((module, centers, row_end, column_end))

        owner = np.zeros((n_rows, n_columns), dtype=np.bool_)
        for module, centers, row_end, column_end in placements:
            rows = slice(module.row_offset, row_end)
            columns = slice(module.column_offset, column_end)
            if owner[rows, columns].any():
                raise ValueError("detector modules overlap in the dense data layout")
            owner[rows, columns] = True

        dense_centers = np.zeros(
            (len(source), n_rows, n_columns, 3), dtype=np.float32)
        dense_mask = np.zeros((n_rows, n_columns), dtype=np.bool_)
        module_slices = []
        for module, centers, row_end, column_end in placements:
            rows = slice(module.row_offset, row_end)
            columns = slice(module.column_offset, column_end)
            dense_centers[:, rows, columns, :] = centers
            dense_mask[rows, columns] = True
            module_slices.append((rows, columns))

        result = cls(source, dense_centers, dense_mask)
        result._module_slices = tuple(module_slices)
        return result

    def pack_module_data(self, module_data, fill_value=0.0):
        """Pack per-module ``(views, rows, columns)`` values into the dense layout."""
        if self._module_slices is None:
            raise ValueError("pack_module_data is available only for module geometry")
        if len(module_data) != len(self._module_slices):
            raise ValueError("module_data count must match the module geometry")
        try:
            fill = float(fill_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("fill_value must be a finite scalar") from exc
        if not np.isfinite(fill):
            raise ValueError("fill_value must be a finite scalar")
        packed = np.full(self.valid_mask.shape, fill, dtype=np.float32)
        for data, (rows, columns) in zip(module_data, self._module_slices):
            values = np.asarray(data)
            expected = (len(self.source_positions_mm),
                        rows.stop - rows.start, columns.stop - columns.start)
            if values.shape != expected:
                raise ValueError(
                    f"module data must have shape {expected}; got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError("module data must contain only finite values")
            packed[:, rows, columns] = values
        return packed

    def unpack_module_data(self, dense_data):
        """Extract a dense ``(views, rows, columns)`` array into module arrays."""
        if self._module_slices is None:
            raise ValueError("unpack_module_data is available only for module geometry")
        values = np.asarray(dense_data)
        if values.shape != self.valid_mask.shape:
            raise ValueError(
                f"dense_data must have shape {self.valid_mask.shape}; got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("dense_data must contain only finite values")
        return [np.ascontiguousarray(values[:, rows, columns])
                for rows, columns in self._module_slices]


def linear_source_surface(pixel_centers_mm, source_start_mm, source_end_mm, n_views, valid_mask=None):
    """Surface for a stationary detector and a source moving along a straight line.

    pixel_centers_mm: ``(rows, cols, 3)`` fixed pixel centres (any shape: flat, curved,
    modular). The source takes ``n_views`` equally spaced positions from
    ``source_start_mm`` to ``source_end_mm`` (world x, y, z). Only the iterative
    ``*_surface`` solvers apply to this limited-angle geometry; ``fdk_surface`` does not.
    """
    pixels = np.asarray(pixel_centers_mm, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("pixel_centers_mm must have shape (rows, cols, 3)")
    if n_views < 1:
        raise ValueError("n_views must be at least 1")
    t = np.linspace(0.0, 1.0, int(n_views))[:, None]
    start = np.asarray(source_start_mm, dtype=np.float64)
    end = np.asarray(source_end_mm, dtype=np.float64)
    source = start + t * (end - start)
    mask = None
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
    # (1, rows, cols, 3) is broadcast to every view by the constructor
    return DetectorSurfaceGeometry(source, np.broadcast_to(pixels, (int(n_views),) + pixels.shape), mask)


def surface_from_geometry(geo, angles):
    """Express a standard TIGRE cone-beam geometry as a DetectorSurfaceGeometry.

    Reproduces the kernels' convention exactly: detector roll/pitch/yaw, then
    detector offsets, then the gantry rotation (alpha, theta, psi), then COR.
    ``angles`` is ``(n,)`` (rotation about z) or ``(n, 3)``. The result can be
    used as a reference or as the starting point for editing individual pixels.
    """
    import copy

    geox = copy.deepcopy(geo)
    angles = np.asarray(angles, dtype=np.float64)
    geox.check_geo(angles.astype(np.float32))
    n_views = len(geox.angles)
    ang = np.asarray(geox.angles, dtype=np.float64).reshape(n_views, -1)
    alpha = ang[:, 0]
    theta = ang[:, 1] if ang.shape[1] > 1 else np.zeros(n_views)
    psi = ang[:, 2] if ang.shape[1] > 2 else np.zeros(n_views)

    n_rows, n_cols = (int(v) for v in geox.nDetector)
    d_v, d_u = (float(v) for v in geox.dDetector)
    u = d_u * (np.arange(n_cols) - (n_cols - 1) / 2.0)
    v = d_v * (np.arange(n_rows) - (n_rows - 1) / 2.0)
    local = np.zeros((n_rows, n_cols, 3))
    local[..., 1] = u[None, :]
    local[..., 2] = v[:, None]

    DSO = np.asarray(geox.DSO, dtype=np.float64)
    DSD = np.asarray(geox.DSD, dtype=np.float64)
    off_det = np.asarray(geox.offDetector, dtype=np.float64)
    rot_det = np.asarray(geox.rotDetector, dtype=np.float64)
    cor = np.asarray(geox.COR, dtype=np.float64)

    source = np.zeros((n_views, 3))
    pixels = np.zeros((n_views, n_rows, n_cols, 3))
    for i in range(n_views):
        yaw, pitch, roll = rot_det[i, 0], rot_det[i, 1], rot_det[i, 2]
        cy, sy, cp, sp, cr, sr = (np.cos(yaw), np.sin(yaw), np.cos(pitch),
                                  np.sin(pitch), np.cos(roll), np.sin(roll))
        r_det = np.array([
            [cr * cp, cr * sp * sy - sr * cy, cr * sp * cy + sr * sy],
            [sr * cp, sr * sp * sy + cr * cy, sr * sp * cy - cr * sy],
            [-sp, cp * sy, cp * cy]])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        ct, st = np.cos(theta[i]), np.sin(theta[i])
        cs, ss = np.cos(psi[i]), np.sin(psi[i])
        r_gantry = np.array([
            [ca * ct * cs - sa * ss, -ca * ct * ss - sa * cs, ca * st],
            [sa * ct * cs + ca * ss, -sa * ct * ss + ca * cs, sa * st],
            [-st * cs, st * ss, ct]])
        shift = np.array([-cor[i] * sa, cor[i] * ca, 0.0])
        det = local @ r_det.T
        det[..., 0] -= DSD[i] - DSO[i]
        det[..., 1] += off_det[i, 1]
        det[..., 2] += off_det[i, 0]
        pixels[i] = det @ r_gantry.T + shift
        source[i] = r_gantry @ np.array([DSO[i], 0.0, 0.0]) + shift
    return DetectorSurfaceGeometry(source, pixels)
