"""Backward-compatibility shim: the tilted-axis utilities merged into
common_geometry (2026-09-02). Import from tigre.utilities.common_geometry.
"""
from .common_geometry import (  # noqa: F401
    tilted_axis_geo,
    project_points_tilted,
    _tilt_matrix,
    _lab_entities,
)
