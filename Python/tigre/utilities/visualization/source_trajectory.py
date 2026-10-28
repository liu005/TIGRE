"""Source paths in the translating object's coordinate frame."""

import numpy as np


class SourceTrajectoryRenderer:
    """Plot world-xyz source positions minus per-view volume-center offsets."""

    def __init__(self, ax, surface, geometry=None):
        source = np.asarray(surface.source_positions_mm, dtype=float)
        offsets = np.atleast_2d(np.asarray(
            getattr(geometry, "offOrigin", np.zeros(3)), dtype=float))
        if (offsets.ndim != 2 or offsets.shape[1] != 3
                or len(offsets) not in (1, len(source)) or not np.isfinite(offsets).all()):
            raise ValueError("offOrigin must contain one finite z/y/x offset or one per source view.")
        self.positions = source - offsets[:, ::-1]
        self.axes = ax
        ax.plot(*self.positions.T, color="mediumpurple", linewidth=1.2)
        ax.scatter(0, 0, 0, color="black", marker="+", s=45)
        ax.scatter(*self.positions[0], color="green", s=18)
        ax.scatter(*self.positions[-1], color="gray", s=18)
        self.marker, = ax.plot([], [], [], marker="o", color="crimson", markersize=5)
        ax.set_xlabel("X (mm)", fontsize=8, labelpad=0)
        ax.set_ylabel("Y (mm)", fontsize=8, labelpad=0)
        ax.set_zlabel("Z (mm)", fontsize=8, labelpad=0)
        ax.tick_params(labelsize=7, pad=0)
        # Keep X/Y on the same scale; expand Z independently to reveal small pitches.
        lo = np.minimum(self.positions.min(axis=0), 0)
        hi = np.maximum(self.positions.max(axis=0), 0)
        center = (lo + hi) / 2
        span = np.maximum(hi - lo, 1)
        span[:2] = max(span[:2])
        for setter, mid, width in zip((ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d), center, span):
            setter(mid - width * .55, mid + width * .55)
        ax.set_box_aspect((1, 1, .8))
        self.update(0)

    def update(self, frame):
        point = self.positions[frame]
        self.marker.set_data_3d(*point[:, None])
        self.axes.set_title(f"Source relative to object\nView {frame}", fontsize=9)
        return self.marker,
