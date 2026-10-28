"""Shared detector-surface scene for script animation and GUI inspection."""
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d

def _detector_runs(surface):
    """Modules (or, without module info, contiguous valid column runs) as (start, stop)."""
    cols = surface.valid_mask.any(axis=(0, 1))
    slices = getattr(surface, "_module_slices", None)
    if slices:
        return [(c.start, c.stop) for _, c in slices if cols[c.start:c.stop].any()]
    runs, start = [], None
    for c, ok in enumerate(np.append(cols, False)):
        if ok and start is None:
            start = c
        elif not ok and start is not None:
            runs.append((start, c))
            start = None
    return runs


def _surface_mesh(surface, view, max_rows=24, max_columns=48):
    """Sample cells within module bounds; never span a masked pixel or module join.

    Vertices are measured pixel centres, not inferred physical pixel edges.
    Single-row/column detectors are represented by line segments.
    """
    pixels = surface.pixel_centers_mm[view]
    mask = surface.valid_mask[view]
    modules = getattr(surface, "_module_slices", None)
    if not modules:
        modules = [(slice(0, mask.shape[0]), slice(c0, c1))
                   for c0, c1 in _detector_runs(surface)]
    faces, segments = [], []
    for rows, columns in modules:
        r0, r1, c0, c1 = rows.start, rows.stop, columns.start, columns.stop
        rr = np.unique(np.linspace(r0, r1 - 1, min(r1 - r0, max_rows), dtype=int))
        cc = np.unique(np.linspace(c0, c1 - 1, min(c1 - c0, max_columns), dtype=int))
        if len(rr) == 1 or len(cc) == 1:
            indices = ([(rr[0], c) for c in cc] if len(rr) == 1
                       else [(r, cc[0]) for r in rr])
            for (ra, ca), (rb, cb) in zip(indices, indices[1:]):
                if mask[ra:rb + 1, ca:cb + 1].all():
                    segments.append([pixels[ra, ca], pixels[rb, cb]])
        else:
            for ra, rb in zip(rr, rr[1:]):
                for ca, cb in zip(cc, cc[1:]):
                    if mask[ra:rb + 1, ca:cb + 1].all():
                        # Triangles also represent curved/non-planar grids reliably.
                        p00, p01 = pixels[ra, ca], pixels[ra, cb]
                        p10, p11 = pixels[rb, ca], pixels[rb, cb]
                        faces.extend([[p00, p01, p11], [p00, p11, p10]])
    return faces, segments


def _volume_box(geo):
    """Validate volume metadata and return initial world xyz bounds."""
    size = np.asarray(geo.sVoxel, dtype=float)
    offsets = np.atleast_2d(np.asarray(geo.offOrigin, dtype=float))
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("sVoxel must contain three positive finite values.")
    if offsets.ndim != 2 or offsets.shape[1] != 3 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("offOrigin must contain finite three-coordinate offsets.")
    return offsets[0][::-1], size[::-1]


class SurfaceRenderer:
    """Build a scene on existing 3D axes and update its artists by view index.

    ``geo`` optionally supplies the volume. ``panel_vertices`` supplies physical
    panel outlines for builder geometry; arbitrary loaded surfaces use a mesh.
    Coordinates are displayed as supplied, with no extra rotation/COR applied.
    """

    def __init__(self, ax, surface, geo=None, pos=0, max_points=4000,
                 world_axes=True, panel_vertices=None):
        src = np.asarray(surface.source_positions_mm)
        pix = np.asarray(surface.pixel_centers_mm)
        mask = np.asarray(surface.valid_mask)
        n_views = len(src)
        pos = 0 if pos is None or pos < 0 or pos >= n_views else int(pos)
        if not mask.any():
            raise ValueError("the detector surface has no valid pixels")

        origin, size = (np.zeros(3), np.zeros(3)) if geo is None else _volume_box(geo)
        volume_mesh = None
        origins = np.zeros((1, 3)) if geo is None else np.atleast_2d(geo.offOrigin)[:, ::-1]
        if len(origins) not in (1, n_views):
            raise ValueError("offOrigin must contain one offset or one offset per view.")
        if geo is not None:
            from .animate_geometry import calCube
            volume_mesh = art3d.Poly3DCollection(calCube(origin, size), color="c", alpha=0.2)
            ax.add_collection3d(volume_mesh)
        if world_axes:
            from .geometry_markers import draw_world_axes
            length = (max(float(np.min(size)) / 2, 1) if geo is not None
                      else max(float(np.linalg.norm(src, axis=1).max()), 1) * 0.12)
            draw_world_axes(ax, length)

        ax.plot3D(*src.T, color="crimson", linestyle="--", alpha=0.5, label="Source path")
        source = ax.scatter(*src[pos], color="crimson", s=35, label="Source")
        source_label = ax.text(*src[pos], "S")
        mesh = art3d.Poly3DCollection([], facecolors="cornflowerblue", edgecolors="navy",
                                     linewidths=0.2, alpha=0.35)
        lines = art3d.Line3DCollection([], colors="navy", linewidths=1.5)
        ax.add_collection3d(mesh, autolim=False)
        ax.add_collection3d(lines, autolim=False)
        cloud = ax.scatter([], [], [], s=1, color="navy", alpha=0.6)
        central_ray, = ax.plot([], [], [], color="pink")
        title = ax.set_title("")

        # Scan frame-wise to avoid copying a large acquisition into float64.
        lo = np.minimum(src.min(axis=0), np.minimum(origin - size / 2, 0))
        hi = np.maximum(src.max(axis=0), np.maximum(origin + size / 2, 0))
        lo = np.minimum(lo, origins.min(axis=0) - size / 2)
        hi = np.maximum(hi, origins.max(axis=0) + size / 2)
        module_count = len(_detector_runs(surface))
        for frame in range(n_views):
            valid = pix[frame][mask[frame]]
            if len(valid):
                lo, hi = np.minimum(lo, valid.min(axis=0)), np.maximum(hi, valid.max(axis=0))
        span = np.maximum(hi - lo, 1)
        # Leave vertical breathing room for thin detectors, preserving mm scale.
        middle_z = (lo[2] + hi[2]) / 2
        span[2] = max(span[2], 0.5 * max(span[0], span[1]))
        lo[2], hi[2] = middle_z - span[2] / 2, middle_z + span[2] / 2
        padding = span * 0.04
        for setter, low, high in zip((ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d), lo-padding, hi+padding):
            setter(low, high)
        ax.set_box_aspect(span)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

        def update(frame):
            if volume_mesh is not None:
                volume_mesh.set_verts(calCube(origins[0 if len(origins) == 1 else frame], size))
            source._offsets3d = tuple(src[frame, :, None])
            source_label.set_position_3d(src[frame])
            if panel_vertices is None:
                faces, segments = _surface_mesh(surface, frame)
            else:
                faces, segments = panel_vertices(frame), []
            mesh.set_verts(faces)
            lines.set_segments(segments)
            valid = pix[frame][mask[frame]]
            stride = max(1, int(np.ceil(len(valid) / max_points)))
            samples = valid[::stride]
            cloud._offsets3d = tuple(samples.T)
            if len(valid):
                centre = valid.mean(axis=0, dtype=np.float64)
                central_ray.set_data_3d(*np.stack((src[frame], centre)).T)
            else:
                central_ray.set_data_3d([], [], [])
            title.set_text(f"Detector surface geometry - view {frame}/{n_views}\n"
                           f"{module_count} detector module(s), {len(valid)} valid pixels")
            artists = (source, source_label, mesh, lines, cloud, central_ray, title)
            return artists + ((volume_mesh,) if volume_mesh is not None else ())

        self.surface = surface
        self.mesh = mesh
        self.source = source
        self.volume = volume_mesh
        self.update = update
        self.update(pos)
