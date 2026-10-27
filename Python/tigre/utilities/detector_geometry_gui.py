"""Interactive desktop builder for modular detector surfaces."""

import sys
import queue
import threading
from pathlib import Path

import numpy as np

MAX_GUI_VIEWS = 6000
MAX_MODULE_DIMENSION = 4096
MAX_MODULES = 256
MAX_TOTAL_PIXELS = 20_000_000

if __package__:
    from .detector_geometry import (
        cylindrical_modular_detector_surface,
        linear_source_surface,
        piecewise_linear_modular_detector_surface,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tigre.utilities.detector_geometry import (
        cylindrical_modular_detector_surface,
        linear_source_surface,
        piecewise_linear_modular_detector_surface,
    )

DETECTOR_FIELD_HELP = {
    "Detector motion": (
        "Toggle whether the detector rotates with the source or remains fixed. "
        "Fixed modes use a straight source trajectory or a circular source arc. "
        "Use iterative reconstruction for limited-angle scans."
    ),
    "Layout": (
        "Choose how module centers are arranged. Cylindrical places panels along "
        "an arc; Piecewise linear lets you specify panel centers and angles, "
        "including a U-shaped detector."
    ),
    "Views": (
        "Number of source positions. For rotating source + detector, positions "
        "span the selected scan angle, also in fixed-arc mode. Fixed-linear mode "
        "samples the start-to-end line equally."
    ),
    "Orbit start (degrees)": (
        "Starting source angle about +z for a rotating source and detector. "
        "Used in rotating and fixed-arc modes; ignored in fixed-linear mode."
    ),
    "Scan span (degrees)": (
        "Angular distance travelled by the rotating source and detector. A "
        "360-degree scan omits the duplicate endpoint; shorter scans include "
        "both endpoints. Ignored when the detector is stationary."
    ),
    "Source radius (mm)": (
        "Radius of the source orbit when the detector moves with the source. "
        "Used in rotating and fixed-arc modes; ignored in fixed-linear mode."
    ),
    "Source height (mm)": (
        "Source z coordinate for rotating and fixed-arc orbits. In fixed-linear mode, "
        "the source start and end coordinates are set explicitly instead."
    ),
    "Source start (x, y, z mm)": (
        "Start point of the source's straight-line trajectory in world x, y, z "
        "coordinates. Used only when the detector is stationary."
    ),
    "Source end (x, y, z mm)": (
        "End point of the source's straight-line trajectory in world x, y, z "
        "coordinates. Used only when the detector is stationary."
    ),
    "COR (mm)": (
        "One global TIGRE centre-of-rotation correction for the rotating scan. "
        "Shifts the source and all panels together along rotating local +y. "
        "The field stays editable in every mode; stationary detectors require zero. "
        "For individual panel alignment, use panel offsets and tilt, not COR."
    ),
    "Panel roll / x (degrees)": "Rotation about panel local x (normal), then local y, then the existing z angle. One value for all panels or one per panel.",
    "Panel pitch / y (degrees)": "Rotation about panel local y (column direction). Tilts about the panel centre. One value for all panels or one per panel.",
    "Module rows": (
        "Number of pixel rows in each panel. All panels have the same row count."
    ),
    "Module columns": (
        "Number of pixel columns in each panel. All panels have the same column count."
    ),
    "Row pitch (mm)": (
        "Center-to-center spacing between adjacent pixel rows, in millimetres."
    ),
    "Column pitch (mm)": (
        "Center-to-center spacing between adjacent pixel columns, in millimetres."
    ),
    "Detector radius (mm)": (
        "Distance from the rotation axis to the nominal detector plane, in millimetres. "
        "For a cylindrical layout this is the arc radius."
    ),
    "Gap columns": (
        "Number of absent data columns inserted between neighboring panels. "
        "These samples are masked in the returned surface."
    ),
    "Modules (cylindrical)": (
        "Number of equal-sized panels distributed along the cylindrical arc. "
        "Used only for the Cylindrical layout."
    ),
    "Panels (from lists)": (
        "Piecewise-linear panel count is inferred from the number of entries in "
        "the transverse-offset, normal-offset, angle, and z-offset lists."
    ),
    "Arc span (degrees)": (
        "Angle from the first to the last panel-center radial direction. "
        "Panels are tangent to the cylinder. Used only for the Cylindrical layout."
    ),
    "Panel transverse offsets (mm)": (
        "Comma-separated transverse (local +y) offsets of the panel centers at "
        "view zero. Used only for Piecewise linear; list one value per panel."
    ),
    "Panel normal offsets (mm)": (
        "Comma-separated offsets along local +x from the nominal detector plane "
        "center at (-detector radius, 0, 0). Positive moves toward the source at view zero. "
        "Used only for Piecewise linear."
    ),
    "Panel yaw / z (degrees)": (
        "Comma-separated panel rotations about +z at view zero. Zero panels face "
        "the source; positive angles rotate counterclockwise viewed from +z. "
        "Used only for Piecewise linear."
    ),
    "Panel z offsets (mm)": (
        "Comma-separated vertical offsets of panel centers from the nominal "
        "detector center. Used only for Piecewise linear."
    ),
}


# Overrides of the GUI defaults; each selection resets all fields first.
GUI_EXAMPLES = {
    "Circular CT: curved modular detector": ("Cylindrical", False, {},
        "Full orbit with eight curved modules and physical gaps. Use Play to inspect the rotating rig."),
    "Short scan: curved detector": ("Cylindrical", False,
        {"scan_span_deg": "240", "angular_span_deg": "40", "column_gap": "0"},
        "A 240-degree orbit with a narrower fan. Inspect the compatibility report before choosing FDK."),
    "TIGRE: circular cone-beam CT": ("Piecewise linear", False,
        {"n_views": "50", "orbit_start_deg": "0", "scan_span_deg": "360",
         "source_radius_mm": "1000", "source_z_mm": "0",
         "detector_radius_mm": "536", "rows": "128", "columns": "128",
         "row_pitch_mm": "3.2", "column_pitch_mm": "3.2", "column_gap": "0",
         "transverse_centers_mm": "0", "normal_offsets_mm": "0",
         "panel_angles_deg": "0", "z_offsets_mm": "0"},
        "TIGRE low-resolution cone geometry: 128 x 128 flat panel, DSO 1000 mm, DSD 1536 mm, full circular orbit."),
    "Three tilted panels": ("Piecewise linear", False,
        {"rows": "256", "columns": "256",
         "transverse_centers_mm": "-256, 0, 256"},
        "Three independently tilted 256 x 256 mm panels with gaps. Inspect panel joins and the FDK compatibility report."),
    "Staggered panels": ("Piecewise linear", False,
        {"rows": "256", "columns": "256",
         "transverse_centers_mm": "-256, 0, 256",
         "z_offsets_mm": "-96, 0, 96"},
        "Three 256 x 256 mm panels at different heights demonstrate a non-extruded surface. Use surface iterative reconstruction."),
    "Tomosynthesis: arc source, fixed panel": ("Piecewise linear", "arc",
        {"rows": "256", "columns": "512", "column_gap": "0",
         "transverse_centers_mm": "0", "normal_offsets_mm": "0",
         "panel_angles_deg": "0", "z_offsets_mm": "0",
         "orbit_start_deg": "-30", "scan_span_deg": "60", "n_views": "61"},
        "Fixed 512 x 256 mm panel with a source tracing a 60-degree circular arc about z. Use iterative reconstruction."),
    "Tomosynthesis: linear source, fixed panel": ("Piecewise linear", True,
        {"transverse_centers_mm": "0", "normal_offsets_mm": "0",
         "panel_angles_deg": "0", "z_offsets_mm": "0", "rows": "256", "columns": "512",
         "source_start_mm": "1000, -500, 0", "source_end_mm": "1000, 500, 0",
         "column_gap": "0", "n_views": "41"},
        "A source translates 1000 mm, from y = -500 to +500 mm, past a fixed 512 x 256 mm flat panel. Play shows the limited-angle trajectory; use iterative reconstruction."),
    "Tomosynthesis: three fixed panels, linear source": ("Piecewise linear", True,
        {"rows": "256", "columns": "256", "column_gap": "2",
         "transverse_centers_mm": "-256, 0, 256",
         "normal_offsets_mm": "0, 10, 0",
         "panel_angles_deg": "45, 0, -45", "z_offsets_mm": "0, 0, 0",
         "source_start_mm": "1000, -400, 0", "source_end_mm": "1000, 400, 200",
         "n_views": "41"},
        "Three fixed tilted 256 x 256 mm panels while the source translates along a straight line. Use Play to inspect the motion; use surface iterative reconstruction."),
}


# Illustrative constant panel tilt about all three axes.
GUI_EXAMPLES["3D detector tilt: x, y, z"] = (
    "Piecewise linear", False,
    {**GUI_EXAMPLES["TIGRE: circular cone-beam CT"][2],
     "panel_x_angles_deg": "20",
     "panel_y_angles_deg": "15",
     "panel_angles_deg": "10", "cor_mm": "0"},
    "Panel tilt about all three axes: x = 20, y = 15, z = 10 degrees, held constant during a circular scan. Use iterative reconstruction.")
GUI_EXAMPLES["TIGRE: global COR offset"] = (
    "Piecewise linear", False,
    {**GUI_EXAMPLES["TIGRE: circular cone-beam CT"][2],
     "panel_x_angles_deg": "0", "panel_y_angles_deg": "0", "cor_mm": "150"},
    "TIGRE circular cone geometry with an exaggerated +150 mm global COR for visibility; source and detector shift together. COR is not per panel.")


# Keep the large GCT acquisition last in the example menu.
_gct_half = np.arctan2(2.6172 * (64 / 2 - 0.5), 10600.0)
_gct_theta = np.linspace(-_gct_half * 53, _gct_half * 53, 53)
GUI_EXAMPLES["GCT: 53-panel circular scan"] = (
    "Piecewise linear", False,
    {"n_views": "5701", "orbit_start_deg": "156", "scan_span_deg": "228",
     "source_radius_mm": "6100", "source_z_mm": "0",
     "detector_radius_mm": "4500", "rows": "1", "columns": "64",
     "row_pitch_mm": "1", "column_pitch_mm": "2.6172", "column_gap": "0",
     "transverse_centers_mm": ", ".join(f"{v:.12g}" for v in 10600 * np.sin(_gct_theta)),
     "normal_offsets_mm": ", ".join(f"{v:.12g}" for v in 10600 * (1 - np.cos(_gct_theta))),
     "panel_angles_deg": ", ".join(f"{v:.12g}" for v in np.rad2deg(-_gct_theta)),
     "z_offsets_mm": ", ".join(["0"] * 53),
     "panel_x_angles_deg": "0", "panel_y_angles_deg": "0", "cor_mm": "0"},
    "GCT idealized 53-panel arc, 5701 views over 228 degrees. Large geometry (~240 MiB arrays); measured-data pixel corrections are not included.")


def build_modular_surface(
        layout, n_views, source_radius_mm, source_z_mm, module_shape,
        pixel_spacing_mm, detector_radius_mm, column_gap, n_modules,
        angular_span_deg, transverse_centers_mm, normal_offsets_mm,
        panel_angles_deg, z_offsets_mm,
        source_motion="Rotating source + detector",
        source_end_mm=(1000.0, 200.0, 0.0),
        source_start_mm=None, orbit_start_deg=0.0, scan_span_deg=360.0, panel_x_angles_deg=(0.0,),
        panel_y_angles_deg=(0.0,), cor_mm=0.0):
    """Build a surface using distances in mm and angles in degrees.

    Panel x/y angles accept one value or one per panel, constant per scan.
    Rotation about each panel centre is Rz(gantry + panel_z) @ Ry(y) @ Rx(x).
    Cylindrical panel_z angles come from the arc layout. Rotating-mode COR
    translates the shared source and every panel by
    cor_mm * (-sin(gantry), cos(gantry), 0). It is one global scalar;
    use panel position offsets and tilts for individual panel alignment.
    Returned coordinates include these corrections; do not apply them again.
    Fixed panels face the initial linear-source position or the arc midpoint.
    """
    if not isinstance(n_views, (int, np.integer)) or n_views < 1:
        raise ValueError("Number of views must be a positive integer.")
    if isinstance(n_views, (bool, np.bool_)):
        raise ValueError("Number of views must be a positive integer.")
    rows, columns = _checked_module_shape(module_shape)
    panel_count = _panel_count_for_layout(
        layout, n_modules, transverse_centers_mm)
    validate_gui_size(n_views, rows, columns, panel_count, column_gap)
    if source_motion in ("Rotating source + detector", "Fixed detector, arc source"):
        if not np.isfinite(source_radius_mm) or source_radius_mm <= 0:
            raise ValueError("Source radius must be a positive finite number.")
        if not np.isfinite(source_z_mm):
            raise ValueError("Source height must be finite.")
        if not np.isfinite(orbit_start_deg):
            raise ValueError("Orbit start angle must be finite.")
        if not np.isfinite(scan_span_deg) or not 0.0 < scan_span_deg <= 360.0:
            raise ValueError("Scan span must be greater than 0 and at most 360 degrees.")
        source_start = np.array(
            [source_radius_mm, 0.0, source_z_mm], dtype=np.float64)
        endpoint = scan_span_deg < 360.0
        angles = np.deg2rad(orbit_start_deg) + np.linspace(
            0.0, np.deg2rad(scan_span_deg), int(n_views), endpoint=endpoint)
        source = np.column_stack((
            source_radius_mm * np.cos(angles),
            source_radius_mm * np.sin(angles),
            np.full(int(n_views), source_z_mm),
        ))
    elif source_motion == "Fixed detector, moving source":
        if source_start_mm is None:
            source_start_mm = (source_radius_mm, 0.0, source_z_mm)
        source_start = np.asarray(source_start_mm, dtype=np.float64)
        if source_start.shape != (3,) or not np.isfinite(source_start).all():
            raise ValueError("Source start position must contain finite x, y, z coordinates.")
        source_end = np.asarray(source_end_mm, dtype=np.float64)
        if source_end.shape != (3,) or not np.isfinite(source_end).all():
            raise ValueError("Source end position must contain finite x, y, z coordinates.")
        orientation_source = source_start.copy()
        if np.hypot(orientation_source[0], orientation_source[1]) == 0:
            orientation_source[0] = 1.0
        source = orientation_source[None, :]
    else:
        raise ValueError(
            "Source motion must be 'Rotating source + detector' or "
            "'Fixed detector, moving source'.")

    arc_source = None
    if source_motion == "Fixed detector, arc source":
        arc_source = source.copy()
        # Build the fixed panel facing the middle of the acquisition arc.
        middle = np.deg2rad(orbit_start_deg + scan_span_deg / 2)
        source = np.array([[source_radius_mm * np.cos(middle),
                            source_radius_mm * np.sin(middle), source_z_mm]])

    if layout == "Cylindrical":
        detector = cylindrical_modular_detector_surface(
            source,
            module_shape=module_shape,
            pixel_spacing_mm=pixel_spacing_mm,
            detector_radius_mm=detector_radius_mm,
            n_modules=n_modules,
            angular_span_rad=np.deg2rad(angular_span_deg),
            column_gap=column_gap,
        )
    elif layout == "Piecewise linear":
        arrays = (
            np.asarray(normal_offsets_mm, dtype=np.float64),
            np.asarray(transverse_centers_mm, dtype=np.float64),
            np.asarray(z_offsets_mm, dtype=np.float64),
        )
        if not arrays[0].size or any(values.ndim != 1 for values in arrays):
            raise ValueError("Enter one normal, transverse, and z offset per panel.")
        if len({len(values) for values in arrays}) != 1:
            raise ValueError("Normal, transverse, and z offset lists must have equal lengths.")
        panel_angles = np.asarray(panel_angles_deg, dtype=np.float64)
        if panel_angles.ndim != 1 or len(panel_angles) != len(arrays[0]):
            raise ValueError("Enter one panel angle per panel.")
        centers = np.column_stack(arrays)
        detector = piecewise_linear_modular_detector_surface(
            source,
            module_shape=module_shape,
            pixel_spacing_mm=pixel_spacing_mm,
            detector_radius_mm=detector_radius_mm,
            module_centers_mm=centers,
            panel_angles_rad=np.deg2rad(panel_angles),
            column_gap=column_gap,
        )
    else:
        raise ValueError("Layout must be 'Cylindrical' or 'Piecewise linear'.")

    # Rotate each panel about its own centre, in its existing local frame.
    x_angles = _panel_tilts(panel_x_angles_deg, panel_count, "Panel x angles")
    y_angles = _panel_tilts(panel_y_angles_deg, panel_count, "Panel y angles")
    azimuths = np.arctan2(source[:, 1], source[:, 0])
    if layout == "Cylindrical":
        z_angles = (np.linspace(np.deg2rad(angular_span_deg) / 2,
                               -np.deg2rad(angular_span_deg) / 2, panel_count)
                    if panel_count > 1 else np.zeros(1))
    else:
        z_angles = np.deg2rad(panel_angles_deg)
    for index in range(panel_count):
        if x_angles[index] == 0 and y_angles[index] == 0:
            continue
        start = index * (columns + column_gap)
        grid = detector.pixel_centers_mm[:, :, start:start + columns]
        for view, azimuth in enumerate(azimuths):
            base = _panel_rotation(azimuth + z_angles[index], 0, 0)
            rotation = _panel_rotation(azimuth + z_angles[index],
                                       x_angles[index], y_angles[index])
            center = grid[view].mean(axis=(0, 1), dtype=np.float64)
            grid[view] = (grid[view] - center) @ base @ rotation.T + center

    cor_mm = float(cor_mm)
    if not np.isfinite(cor_mm):
        raise ValueError("COR must be finite.")
    if source_motion != "Rotating source + detector" and cor_mm != 0:
        raise ValueError("COR is supported only for rotating source + detector.")
    if cor_mm:
        # Same translation as TIGRE's surface_from_geometry convention.
        shift = cor_mm * np.column_stack((-np.sin(azimuths),
                                          np.cos(azimuths), np.zeros(len(source))))
        detector.source_positions_mm += shift
        detector.pixel_centers_mm += shift[:, None, None, :]

    if source_motion == "Fixed detector, moving source":
        fixed = linear_source_surface(
            detector.pixel_centers_mm[0],
            source_start,
            source_end,
            int(n_views),
            valid_mask=detector.valid_mask[0],
        )
        fixed._module_slices = detector._module_slices
        return fixed
    if arc_source is not None:
        from tigre.utilities.detector_geometry import DetectorSurfaceGeometry
        return DetectorSurfaceGeometry(
            arc_source, np.broadcast_to(detector.pixel_centers_mm,
                (int(n_views),) + detector.pixel_centers_mm.shape[1:]).copy(),
            np.broadcast_to(detector.valid_mask,
                (int(n_views),) + detector.valid_mask.shape[1:]).copy(),
            module_slices=detector._module_slices)
    return detector


def _panel_list_values(text, count, allow_shared):
    values = np.atleast_1d(DetectorGeometryBuilder._parse_numbers(text, "Panel list"))
    if values.size == 1 and allow_shared:
        values = np.repeat(values, count)
    if values.size != count:
        raise ValueError(f"Enter exactly {count} panel values.")
    return values


def _panel_tilts(values, count, name):
    values = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or values.size not in (1, count) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain one finite value or one per panel.")
    return np.deg2rad(np.broadcast_to(values, (count,)))


def _panel_rotation(z, x, y):
    """Local Rx, then Ry, then existing panel/gantry Rz (radians)."""
    cz, sz, cx, sx, cy, sy = np.cos(z), np.sin(z), np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def _checked_module_shape(module_shape):
    shape = np.asarray(module_shape)
    if (shape.shape != (2,) or shape.dtype.kind not in "iu"
            or shape.dtype.kind == "b" or np.any(shape <= 0)):
        raise ValueError("Module rows and columns must be positive integers.")
    rows, columns = (int(value) for value in shape)
    if rows > MAX_MODULE_DIMENSION or columns > MAX_MODULE_DIMENSION:
        raise ValueError(
            f"Each module dimension is limited to {MAX_MODULE_DIMENSION} pixels.")
    return rows, columns


def _panel_count_for_layout(layout, n_modules, transverse_centers_mm):
    if layout == "Cylindrical":
        if (not isinstance(n_modules, (int, np.integer))
                or isinstance(n_modules, (bool, np.bool_)) or n_modules < 1):
            raise ValueError("Module count must be a positive integer.")
        count = int(n_modules)
    elif layout == "Piecewise linear":
        count = len(np.atleast_1d(transverse_centers_mm))
        if count < 1:
            raise ValueError("Enter at least one piecewise-linear panel.")
    else:
        raise ValueError("Layout must be 'Cylindrical' or 'Piecewise linear'.")
    if count > MAX_MODULES:
        raise ValueError(f"The GUI supports at most {MAX_MODULES} detector modules.")
    return count


def validate_gui_size(n_views, rows, columns, panel_count, column_gap):
    """Reject GUI inputs whose generated per-pixel geometry is too large."""
    if n_views > MAX_GUI_VIEWS:
        raise ValueError(f"The GUI supports at most {MAX_GUI_VIEWS} views.")
    if (not isinstance(column_gap, (int, np.integer))
            or isinstance(column_gap, (bool, np.bool_)) or column_gap < 0):
        raise ValueError("Gap columns must be a non-negative integer.")
    dense_columns = panel_count * columns + max(0, panel_count - 1) * column_gap
    total_pixels = n_views * rows * dense_columns
    if total_pixels > MAX_TOTAL_PIXELS:
        raise ValueError(
            f"This layout requires {total_pixels:,} detector samples; "
            f"the GUI limit is {MAX_TOTAL_PIXELS:,}. Reduce views or detector size.")


def check_tigre_compatibility(surface):
    """Check surface-projector validity and whether the geometry fits surface FDK."""
    source = np.asarray(surface.source_positions_mm)
    pixels = np.asarray(surface.pixel_centers_mm)
    mask = np.asarray(surface.valid_mask)
    if source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 1:
        raise ValueError("Source positions must have shape (views, 3).")
    if (pixels.ndim != 4 or pixels.shape[0] != source.shape[0]
            or pixels.shape[-1] != 3 or min(pixels.shape[1:3]) < 1):
        raise ValueError("Detector pixels must have shape (views, rows, columns, 3).")
    if mask.shape != pixels.shape[:3] or mask.dtype.kind != "b":
        raise ValueError("Detector valid_mask must be boolean with shape (views, rows, columns).")
    if not np.isfinite(source).all() or not np.isfinite(pixels).all():
        raise ValueError("Source and detector coordinates must be finite.")
    if not mask.any():
        raise ValueError("The detector has no valid pixels.")

    report = {
        "iterative_supported": True,
        "iterative_message": (
            "Supported by Ax_surface/Atb_surface and surface iterative algorithms "
            "(cone-beam Siddon; GPU runtime/build not tested)."
        ),
        "fdk_supported": False,
        "fdk_message": "",
    }
    try:
        from tigre.algorithms.fdk_surface import _analyse

        if source.shape[0] < 2:
            raise ValueError("FDK requires at least two views.")
        _analyse(surface)
    except (ValueError, NotImplementedError) as exc:
        report["fdk_message"] = (
            f"Not supported by the validated circular path: {exc} "
            "An experimental per-view path is available with "
            "fdk_surface(..., allow_approximate=True)."
        )
    else:
        report["fdk_supported"] = True
        report["fdk_message"] = (
            "Supported by fdk_surface's validated circular path "
            "(approximate FDK)."
        )
    return report


from tigre.utilities.geometry_io import (
    surface_archive_arrays, load_geometry_archive, load_geometry_file, save_geometry_yaml,
)


class DetectorGeometryBuilder:
    """Tkinter window for creating, previewing, and saving module layouts."""

    def __init__(self, root=None, geometry=None, angles=None, surface=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.tk = tk
        self.ttk = ttk
        self.root = root if root is not None else tk.Tk()
        self.root.title("TIGRE Modular Detector Builder")
        self.surface = None
        self._inspection_mode = False
        self._inspection_geometry = None
        self._builder_geometry = None
        self._surface_renderer = None
        self._fields = {}
        self._entries = {}
        self._animation_after = None
        self._animation_playing = False
        self._frame_index = 0
        self._panel_collection = None
        self._source_artist = None
        self._panel_count = 0
        self._parameters = None
        self._Poly3DCollection = Poly3DCollection
        self._compatibility_queue = queue.Queue()
        self._compatibility_check_pending = False
        self._compatibility_surface = None
        self._remap_queue = queue.Queue()
        self._remap_pending = False
        self._remap_surface = None
        self._save_button = None
        self._remap_button = None

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(outer)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        preview = ttk.Frame(outer)
        preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        source_controls = ttk.LabelFrame(controls, text="Source", padding=6)
        source_controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        detector_controls = ttk.LabelFrame(controls, text="Detector", padding=6)
        detector_controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        source_controls.columnconfigure(1, weight=1)
        detector_controls.columnconfigure(1, weight=1)

        self._detector_stationary = False
        self._add_help_label(detector_controls, "Detector motion").grid(
            row=0, column=0, sticky="w", pady=2)
        self._detector_motion_button = ttk.Button(
            detector_controls, text="Moving with source", command=self.toggle_detector_motion)
        self._detector_motion_button.grid(row=0, column=1, sticky="ew", pady=2)
        self._detector_motion_button.bind("<Enter>", self._show_help)

        self._layout = tk.StringVar(value="Cylindrical")
        self._add_help_label(detector_controls, "Layout").grid(
            row=1, column=0, sticky="w", pady=2)
        layout_box = ttk.Combobox(
            detector_controls, textvariable=self._layout,
            values=("Cylindrical", "Piecewise linear"), state="readonly", width=20)
        self._layout_box = layout_box
        layout_box.grid(row=1, column=1, sticky="ew", pady=2)
        layout_box.bind("<<ComboboxSelected>>", self._update_module_count_field)

        fields = (
            ("Views", "n_views", "60"),
            ("Orbit start (degrees)", "orbit_start_deg", "0"),
            ("Scan span (degrees)", "scan_span_deg", "360"),
            ("Source radius (mm)", "source_radius_mm", "1000"),
            ("Source height (mm)", "source_z_mm", "0"),
            ("Source start (x, y, z mm)", "source_start_mm", "1000, 0, 0"),
            ("Source end (x, y, z mm)", "source_end_mm", "1000, 200, 0"),
            ("COR (mm)", "cor_mm", "0"),
            ("Panel roll / x (degrees)", "panel_x_angles_deg", "0"),
            ("Panel pitch / y (degrees)", "panel_y_angles_deg", "0"),
            ("Module rows", "rows", "64"),
            ("Module columns", "columns", "32"),
            ("Row pitch (mm)", "row_pitch_mm", "1"),
            ("Column pitch (mm)", "column_pitch_mm", "1"),
            ("Detector radius (mm)", "detector_radius_mm", "600"),
            ("Gap columns", "column_gap", "2"),
            ("Modules (cylindrical)", "n_modules", "8"),
            ("Arc span (degrees)", "angular_span_deg", "120"),
            ("Panel transverse offsets (mm)", "transverse_centers_mm", "-32, 0, 32"),
            ("Panel normal offsets (mm)", "normal_offsets_mm", "0, 10, 0"),
            ("Panel yaw / z (degrees)", "panel_angles_deg", "45, 0, -45"),
            ("Panel z offsets (mm)", "z_offsets_mm", "0, 0, 0"),
        )
        source_fields = {
            "n_views", "orbit_start_deg", "scan_span_deg", "source_radius_mm",
            "source_z_mm", "source_start_mm", "source_end_mm",
        }
        self._panel_count_display = tk.StringVar(value="")
        self._list_buttons = {}
        self._panel_list_fields = {
            "panel_x_angles_deg", "panel_y_angles_deg", "panel_angles_deg",
            "transverse_centers_mm", "normal_offsets_mm", "z_offsets_mm",
        }
        source_row, detector_row = 0, 2
        for label, name, default in fields:
            if name in source_fields:
                parent, row = source_controls, source_row
                source_row += 1
            else:
                parent, row = detector_controls, detector_row
                detector_row += 1
            label_widget = self._add_help_label(parent, label)
            label_widget.grid(
                row=row, column=0, sticky="w", pady=2)
            variable = tk.StringVar(value=default)
            self._fields[name] = variable
            entry = ttk.Entry(parent, textvariable=variable, width=22)
            self._entries[name] = entry
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            if name in self._panel_list_fields:
                button = ttk.Button(parent, text="From list...",
                    command=lambda field=name, title=label: self._show_panel_list(field, title))
                button.grid(row=row, column=2, padx=(4, 0), pady=2)
                self._list_buttons[name] = button
            if name == "n_modules":
                self._module_count_label = label_widget
                self._module_count_entry = entry
        self._default_fields = {name: default for _, name, default in fields}
        self._update_module_count_field()
        self._fields["transverse_centers_mm"].trace_add(
            "write", lambda *_: self._update_module_count_field())

        help_frame = ttk.LabelFrame(controls, text="Parameter help", padding=4)
        help_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self._help_text = tk.Text(
            help_frame, height=5, width=40, wrap=tk.WORD, state=tk.DISABLED)
        self._help_text.pack(fill=tk.X, expand=True)
        self._set_help("Layout")

        buttons = ttk.Frame(controls)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self._preview_button = ttk.Button(buttons, text="Preview", command=self.update_preview)
        self._preview_button.pack(side=tk.LEFT, padx=(0, 4))
        self._save_button = ttk.Button(
            buttons, text="Save surface...", command=self.save_surface)
        self._save_button.pack(side=tk.LEFT)
        self._remap_button = ttk.Button(
            buttons, text="Remap data for FDK...", command=self.remap_for_fdk)
        self._remap_button.pack(side=tk.LEFT, padx=(4, 0))
        self._remap_button.configure(state=tk.DISABLED)
        loading = ttk.Frame(controls)
        loading.grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Button(loading, text="Load geometry...", command=self.load_geometry).pack(side=tk.LEFT)
        self._builder_button = ttk.Button(loading, text="Return to builder", command=self.return_to_builder, state=tk.DISABLED)
        self._builder_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(loading, text="Save geometry YAML...", command=self.save_geometry_settings).pack(side=tk.LEFT)
        self._status = tk.StringVar(value="Set parameters, then select Preview.")
        ttk.Label(
            controls, textvariable=self._status, wraplength=300, justify=tk.LEFT
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        self._figure = Figure(figsize=(8, 7), dpi=100)
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._canvas = FigureCanvasTkAgg(self._figure, master=preview)
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        playback = ttk.Frame(preview)
        playback.pack(fill=tk.X)
        self._play_button = ttk.Button(
            playback, text="Play", command=self.toggle_animation)
        self._play_button.pack(side=tk.LEFT, padx=(0, 8))
        self._frame_slider = tk.Scale(
            playback, from_=0, to=0, orient=tk.HORIZONTAL,
            showvalue=True, label="View", command=self._on_frame_slider)
        self._frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._play_button.configure(state=tk.DISABLED)
        self._frame_slider.configure(state=tk.DISABLED)
        examples = ttk.Frame(preview)
        examples.pack(fill=tk.X)
        ttk.Label(examples, text="Example:").pack(side=tk.LEFT)
        self._example = tk.StringVar(value=next(iter(GUI_EXAMPLES)))
        ttk.Combobox(examples, textvariable=self._example,
                     values=tuple(GUI_EXAMPLES), state="readonly", width=42).pack(side=tk.LEFT)
        ttk.Button(examples, text="Load example", command=self.load_example).pack(side=tk.LEFT)
        self._example_description = tk.StringVar(value="")
        ttk.Label(examples, textvariable=self._example_description,
                  wraplength=420).pack(side=tk.LEFT, padx=8)
        if geometry is not None or surface is not None:
            self.inspect_geometry(geometry, angles=angles, surface=surface)
        else:
            self.update_preview()

    def load_example(self):
        """Reset the controls and preview the selected demonstration."""
        self._inspection_mode = False
        self._builder_geometry = None
        self._builder_button.configure(state=self.tk.DISABLED)
        layout, stationary, overrides, description = GUI_EXAMPLES[self._example.get()]
        self._stop_animation()
        for name, value in self._default_fields.items():
            self._fields[name].set(overrides.get(name, value))
        self._layout.set(layout)
        self._detector_stationary = stationary
        self._detector_motion_button.configure(
            text=("Stationary (arc source)" if stationary == "arc" else
                  "Stationary" if stationary else "Moving with source"))
        self._update_module_count_field()
        self._example_description.set(description)
        self.update_preview()

    def load_geometry(self):
        """Load a numeric NPZ archive; never execute or unpickle geometry files."""
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(parent=self.root, title="Inspect geometry",
            filetypes=(("Geometry files", "*.npz *.yaml *.yml"), ("YAML settings", "*.yaml *.yml"), ("Surface archive", "*.npz")))
        if not path:
            return
        try:
            self.load_geometry_path(path)
        except (ValueError, TypeError, AttributeError, OSError, KeyError, MemoryError, ImportError) as exc:
            messagebox.showerror("Cannot load geometry", str(exc), parent=self.root)

    def load_geometry_path(self, path):
        """Load editable modular YAML or inspect an exact saved surface."""
        document = load_geometry_file(path)
        if document.settings is None:
            self.inspect_geometry(document.geometry, surface=document.surface)
            return
        settings = document.settings
        self._stop_animation()
        self._inspection_mode = False
        self._builder_geometry = document.geometry
        for name, default in self._default_fields.items():
            value = settings.get(name, default)
            if name in ("rows", "columns"):
                value = settings["module_shape"][0 if name == "rows" else 1]
            elif name in ("row_pitch_mm", "column_pitch_mm"):
                spacing = np.atleast_1d(settings["pixel_spacing_mm"])
                value = spacing[0 if name == "row_pitch_mm" or len(spacing) == 1 else 1]
            if value is None:
                value = default
            if isinstance(value, (list, tuple, np.ndarray)):
                value = ", ".join(str(item) for item in value)
            self._fields[name].set(str(value))
        self._layout.set(settings["layout"])
        self._detector_stationary = {
            "Rotating source + detector": False,
            "Fixed detector, moving source": True,
            "Fixed detector, arc source": "arc",
        }[settings["source_motion"]]
        state = ("Stationary (arc source)" if self._detector_stationary == "arc" else
                 "Stationary" if self._detector_stationary else "Moving with source")
        self._detector_motion_button.configure(text=state)
        self._builder_button.configure(state=self.tk.DISABLED)
        self._update_module_count_field()
        self._example_description.set(f"Loaded editable settings: {Path(path).name}")
        self.update_preview()

    def save_geometry_settings(self):
        """Save current builder controls, or reference exact inspection arrays."""
        from tkinter import filedialog, messagebox
        try:
            inspection = getattr(self, "_inspection_mode", False)
            settings = None if inspection else self._read_parameters()
            if inspection and self.surface is None:
                raise ValueError("No geometry to save.")
            path = filedialog.asksaveasfilename(parent=self.root,
                title="Save geometry YAML", defaultextension=".yaml",
                filetypes=(("YAML geometry", "*.yaml *.yml"),))
            if not path:
                return
            save_geometry_yaml(path, settings=settings,
                surface=self.surface if inspection else None,
                geometry=self._inspection_geometry if inspection else self._builder_geometry)
            self._status.set(f"Saved geometry YAML: {path}")
        except (ValueError, TypeError, OSError, ImportError, MemoryError) as exc:
            messagebox.showerror("Cannot save geometry YAML", str(exc), parent=self.root)

    def inspect_geometry(self, geometry=None, angles=None, surface=None):
        """Inspect a TIGRE geometry plus angles, or an explicit detector surface.

        Construction controls are disabled; coordinates and masks are preserved.
        """
        from tigre.utilities.detector_geometry import DetectorSurfaceGeometry, surface_from_geometry
        from tigre.utilities.visualization.surface_renderer import SurfaceRenderer
        if isinstance(geometry, (str, Path)) and surface is None and angles is None:
            return self.load_geometry_path(geometry)
        if surface is None:
            if geometry is None:
                raise ValueError("Supply a geometry with angles, or a detector surface.")
            if angles is None:
                angles = getattr(geometry, "angles", None)
            if angles is None:
                raise ValueError("Projection angles are required to inspect a TIGRE geometry.")
            if getattr(geometry, "mode", "cone") != "cone":
                raise ValueError("Standard geometry inspection supports cone mode.")
            surface = surface_from_geometry(geometry, np.asarray(angles))
        if not isinstance(surface, DetectorSurfaceGeometry) or not surface.valid_mask.any():
            raise ValueError("Supply a detector surface with valid pixels.")
        if geometry is not None:
            from tigre.utilities.visualization.surface_renderer import _volume_box
            _volume_box(geometry)
        self._stop_animation()
        self._clear_preview()
        self.surface = surface
        self._parameters = None
        self._inspection_mode = True
        self._inspection_geometry = geometry
        self._surface_renderer = SurfaceRenderer(self._axes, surface, geo=geometry)
        self._panel_collection = self._surface_renderer.mesh
        self._source_artist = self._surface_renderer.source
        self._update_field_states()
        self._builder_button.configure(state=self.tk.NORMAL)
        self._play_button.configure(state=self.tk.NORMAL)
        self._frame_slider.configure(state=self.tk.NORMAL, to=len(surface.source_positions_mm) - 1)
        self._remap_button.configure(state=self.tk.NORMAL)
        self._frame_slider.set(0)
        self._render_frame(0)
        self._status.set("Inspection mode: loaded geometry is read-only. Use Return to builder to edit a new layout.")
        self._example_description.set("Inspection mode; builder values are retained but inactive.")
        self._canvas.draw_idle()

    def return_to_builder(self):
        """Restore the retained builder values without approximating loaded data."""
        self._inspection_mode = False
        self._inspection_geometry = None
        self._builder_button.configure(state=self.tk.DISABLED)
        self._update_module_count_field()
        self._example_description.set("")
        self.update_preview()

    def _on_scroll_zoom(self, event):
        """Zoom all three axes together while preserving the viewing angle."""
        if event.inaxes is not self._axes or not event.step:
            return
        factor = 1.2 ** (-float(np.clip(event.step, -10, 10)))
        axes = self._axes
        limits = [axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d()]
        updated = []
        for lower, upper in limits:
            center = (lower + upper) / 2.0
            half_span = (upper - lower) * factor / 2.0
            if not np.isfinite(half_span) or abs(half_span) < 1e-6 or abs(half_span) > 1e12:
                return
            updated.append((center - half_span, center + half_span))
        axes.set_xlim3d(*updated[0])
        axes.set_ylim3d(*updated[1])
        axes.set_zlim3d(*updated[2])
        self._canvas.draw_idle()

    def _add_help_label(self, parent, text):
        label = self.ttk.Label(parent, text=text, cursor="question_arrow")
        label.bind("<Enter>", self._show_help)
        return label

    def _show_help(self, event):
        self._set_help(event.widget.cget("text"))

    def _update_module_count_field(self, event=None):
        if event is not None:
            self._layout.set(event.widget.get())
        if self._layout.get() == "Cylindrical":
            label = "Modules (cylindrical)"
            self._module_count_entry.configure(textvariable=self._fields["n_modules"])
            self._module_count_entry.configure(state=self.tk.NORMAL)
        else:
            label = "Panels (from lists)"
            self._module_count_entry.configure(state=self.tk.DISABLED)
            self._module_count_entry.configure(textvariable=self._panel_count_display)
            try:
                values = self._parse_numbers(self._fields["transverse_centers_mm"].get(), "Panel transverse offsets")
                self._panel_count_display.set(str(np.atleast_1d(values).size))
            except ValueError:
                self._panel_count_display.set("Invalid list")
        self._module_count_label.configure(text=label)
        self._update_field_states()

    def _update_field_states(self):
        """Layout controls panel shape; motion controls source trajectory fields."""
        inspection = getattr(self, "_inspection_mode", False)
        if hasattr(self, "_layout_box"):
            self._layout_box.configure(state="disabled" if inspection else "readonly")
            self._detector_motion_button.configure(state=self.tk.DISABLED if inspection else self.tk.NORMAL)
            if hasattr(self, "_preview_button"):
                self._preview_button.configure(state=self.tk.DISABLED if inspection else self.tk.NORMAL)
        if inspection:
            for widget in list(self._entries.values()) + list(getattr(self, "_list_buttons", {}).values()):
                widget.configure(state=self.tk.DISABLED)
            return
        # Restore shared fields as well as the mode-specific fields below.
        for widget in self._entries.values():
            widget.configure(state=self.tk.NORMAL)
        cylindrical = self._layout.get() == "Cylindrical"
        linear_source = self._detector_stationary is True
        enabled = {
            "cor_mm": True,
            "n_modules": cylindrical,
            "angular_span_deg": cylindrical,
            "transverse_centers_mm": not cylindrical,
            "normal_offsets_mm": not cylindrical,
            "panel_angles_deg": not cylindrical,
            "z_offsets_mm": not cylindrical,
            "source_start_mm": linear_source,
            "source_end_mm": linear_source,
            "source_radius_mm": not linear_source,
            "source_z_mm": not linear_source,
            "orbit_start_deg": not linear_source,
            "scan_span_deg": not linear_source,
        }
        enabled["panel_x_angles_deg"] = True
        enabled["panel_y_angles_deg"] = True
        for name, button in getattr(self, "_list_buttons", {}).items():
            button.configure(state=self.tk.NORMAL if enabled.get(name, True) else self.tk.DISABLED)
        for name, active in enabled.items():
            self._entries[name].configure(
                state=self.tk.NORMAL if active else self.tk.DISABLED)


    def _show_panel_list(self, name, title):
        """Edit panel values in index order, without truncating long lists."""
        from tkinter import messagebox

        try:
            if self._layout.get() == "Cylindrical":
                count = int(self._parse_numbers(self._fields["n_modules"].get(), "Modules", integer=True))
            else:
                count = np.atleast_1d(self._parse_numbers(
                    self._fields["transverse_centers_mm"].get(), "Panel transverse offsets")).size
            if not 1 <= count <= MAX_MODULES:
                raise ValueError(f"Panel count must be between 1 and {MAX_MODULES}.")
            values = _panel_list_values(self._fields[name].get(), count,
                name in ("panel_x_angles_deg", "panel_y_angles_deg"))
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Invalid panel list", str(exc), parent=self.root)
            return
        window = self.tk.Toplevel(self.root)
        window.title(title)
        window.transient(self.root)
        self.ttk.Label(window, text=f"{count} panels: one value per line, in panel order (1 to {count}).").pack(padx=12, pady=8)
        frame = self.ttk.Frame(window)
        frame.pack(fill=self.tk.BOTH, expand=True, padx=12)
        editor = self.tk.Text(frame, width=32, height=min(count + 1, 16), wrap="none")
        scrollbar = self.ttk.Scrollbar(frame, orient="vertical", command=editor.yview)
        editor.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)
        editor.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        editor.insert("1.0", "\n".join(f"{value:.12g}" for value in values))

        def apply():
            try:
                text = editor.get("1.0", self.tk.END).strip().replace("\n", ",")
                updated = _panel_list_values(text, count, False)
            except ValueError as exc:
                messagebox.showerror("Invalid panel list", str(exc), parent=window)
                return
            self._fields[name].set(", ".join(f"{value:.12g}" for value in updated))
            window.destroy()

        buttons = self.ttk.Frame(window)
        buttons.pack(padx=12, pady=10)
        self.ttk.Button(buttons, text="Apply", command=apply).pack(side=self.tk.LEFT, padx=4)
        self.ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side=self.tk.LEFT, padx=4)
        window.grab_set()
        editor.focus_set()

    def toggle_detector_motion(self):
        """Toggle between a stationary detector and one rotating with the source."""
        self._detector_stationary = {False: True, True: "arc", "arc": False}[self._detector_stationary]
        state = ("Stationary (arc source)" if self._detector_stationary == "arc" else
                 "Stationary" if self._detector_stationary else "Moving with source")
        self._detector_motion_button.configure(text=state)
        self._update_field_states()
        self._help_text.configure(state=self.tk.NORMAL)
        self._help_text.delete("1.0", self.tk.END)
        self._help_text.insert("1.0", DETECTOR_FIELD_HELP["Detector motion"])
        self._help_text.insert(self.tk.END, f"\n\nSelected: {state}")
        self._help_text.configure(state=self.tk.DISABLED)
        if self.surface is not None:
            self.update_preview()

    def _set_help(self, name):
        self._help_text.configure(state=self.tk.NORMAL)
        self._help_text.delete("1.0", self.tk.END)
        self._help_text.insert("1.0", DETECTOR_FIELD_HELP[name])
        self._help_text.configure(state=self.tk.DISABLED)

    @staticmethod
    def _parse_numbers(value, name, integer=False):
        try:
            parsed = [float(item.strip()) for item in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of numbers.") from exc
        if not parsed or not np.isfinite(parsed).all():
            raise ValueError(f"{name} must contain finite numbers.")
        if integer:
            if any(item != int(item) for item in parsed):
                raise ValueError(f"{name} must contain integers.")
            return int(parsed[0]) if len(parsed) == 1 else [int(item) for item in parsed]
        return parsed[0] if len(parsed) == 1 else parsed

    def _read_parameters(self):
        values = {name: variable.get().strip() for name, variable in self._fields.items()}
        n_views = int(self._parse_numbers(values["n_views"], "Views", integer=True))
        rows = int(self._parse_numbers(values["rows"], "Module rows", integer=True))
        columns = int(self._parse_numbers(values["columns"], "Module columns", integer=True))
        column_gap = int(self._parse_numbers(values["column_gap"], "Gap columns", integer=True))
        layout = self._layout.get()
        stationary = self._detector_stationary
        if layout == "Cylindrical":
            n_modules = int(self._parse_numbers(
                values["n_modules"], "Modules", integer=True))
            angular_span_deg = float(values["angular_span_deg"])
            transverse, normal, panel_angles, z_offsets = (
                [0.0], [0.0], [0.0], [0.0])
        else:
            n_modules = 1
            angular_span_deg = 0.0
            transverse = self._parse_numbers(
                values["transverse_centers_mm"], "Panel transverse offsets")
            normal = self._parse_numbers(values["normal_offsets_mm"], "Panel normal offsets")
            panel_angles = self._parse_numbers(values["panel_angles_deg"], "Panel angles")
            z_offsets = self._parse_numbers(values["z_offsets_mm"], "Panel z offsets")
        if stationary is True:
            source_start = self._parse_numbers(
                values["source_start_mm"], "Source start position")
            source_end = self._parse_numbers(values["source_end_mm"], "Source end position")
            if not isinstance(source_start, list) or len(source_start) != 3:
                raise ValueError("Source start position must contain x, y, z values.")
            if not isinstance(source_end, list) or len(source_end) != 3:
                raise ValueError("Source end position must contain x, y, z values.")
        else:
            source_start, source_end = None, (1000.0, 200.0, 0.0)
        source_radius = (
            float(values["source_radius_mm"]) if stationary != True else 1000.0)
        source_height = float(values["source_z_mm"]) if stationary != True else 0.0
        orbit_start = (
            float(values["orbit_start_deg"]) if stationary != True else 0.0)
        scan_span = float(values["scan_span_deg"]) if stationary != True else 360.0
        _checked_module_shape((rows, columns))
        panel_count = _panel_count_for_layout(layout, n_modules, transverse)
        validate_gui_size(n_views, rows, columns, panel_count, column_gap)
        return {
            "source_motion": (
                "Fixed detector, arc source" if stationary == "arc" else
                "Fixed detector, moving source"
                if stationary else "Rotating source + detector"),
            "cor_mm": float(values.get("cor_mm", "0")),
            "panel_x_angles_deg": np.atleast_1d(self._parse_numbers(values.get("panel_x_angles_deg", "0"), "Panel x angles")),
            "panel_y_angles_deg": np.atleast_1d(self._parse_numbers(values.get("panel_y_angles_deg", "0"), "Panel y angles")),
            "layout": layout,
            "n_views": n_views,
            "source_radius_mm": source_radius,
            "source_z_mm": source_height,
            "source_start_mm": source_start,
            "source_end_mm": source_end,
            "orbit_start_deg": orbit_start,
            "scan_span_deg": scan_span,
            "module_shape": (rows, columns),
            "pixel_spacing_mm": (
                float(values["row_pitch_mm"]), float(values["column_pitch_mm"])),
            "detector_radius_mm": float(values["detector_radius_mm"]),
            "column_gap": column_gap,
            "n_modules": n_modules,
            "angular_span_deg": angular_span_deg,
            "transverse_centers_mm": np.atleast_1d(transverse),
            "normal_offsets_mm": np.atleast_1d(normal),
            "panel_angles_deg": np.atleast_1d(panel_angles),
            "z_offsets_mm": np.atleast_1d(z_offsets),
        }

    def update_preview(self):
        from tkinter import messagebox

        if getattr(self, "_inspection_mode", False):
            return

        self.surface = None
        self._parameters = None
        self._stop_animation()
        self._clear_preview()
        try:
            parameters = self._read_parameters()
            self.surface = build_modular_surface(**parameters)
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            messagebox.showerror("Invalid detector layout", str(exc), parent=self.root)
            return
        self._parameters = parameters

        axes = self._axes
        axes.clear()
        n_rows, n_columns = parameters["module_shape"]
        if parameters["layout"] == "Cylindrical":
            self._panel_count = parameters["n_modules"]
        else:
            self._panel_count = len(parameters["transverse_centers_mm"])
        panels = self._panel_vertices(parameters, 0)
        from tigre.utilities.visualization.surface_renderer import SurfaceRenderer
        self._surface_renderer = SurfaceRenderer(
            axes, self.surface, geo=getattr(self, "_builder_geometry", None), world_axes=False,
            panel_vertices=lambda frame: self._panel_vertices(parameters, frame))
        self._panel_collection = self._surface_renderer.mesh
        self._source_artist = self._surface_renderer.source
        source_path = self.surface.source_positions_mm
        self._add_coordinate_markers(parameters)
        axes.legend(loc="upper right")
        self._frame_index = 0
        self._play_button.configure(state=self.tk.NORMAL)
        self._frame_slider.configure(state=self.tk.NORMAL)
        self._frame_slider.configure(to=len(source_path) - 1)
        self._remap_button.configure(state=self.tk.NORMAL)
        self._frame_slider.set(0)
        self._render_frame(0)
        self._canvas.draw_idle()
        self._status.set(
            f"{len(panels)} panels, {n_rows} x {n_columns} pixels each; "
            f"dense layout {self.surface.valid_mask.shape[1]} x "
            f"{self.surface.valid_mask.shape[2]} pixels.")

    def _clear_preview(self):
        if hasattr(self, "_axes"):
            self._axes.clear()
            self._canvas.draw_idle()
        self._surface_renderer = None
        self._panel_collection = None
        self._source_artist = None
        self._cor_artist = None
        self._cor_label = None
        self._cor_line = None
        if getattr(self, "_play_button", None) is not None:
            self._play_button.configure(state=self.tk.DISABLED)
        if getattr(self, "_frame_slider", None) is not None:
            self._frame_slider.configure(state=self.tk.DISABLED, to=0)
            self._frame_slider.set(0)
        if getattr(self, "_remap_button", None) is not None:
            self._remap_button.configure(state=self.tk.DISABLED)

    def _add_coordinate_markers(self, parameters):
        """Show the world origin and the actual per-view global COR shift."""
        # Size the triad relative to the rig, while keeping its base at world zero.
        points = self.surface.source_positions_mm
        scale = max(float(np.max(np.linalg.norm(points, axis=1))), 1.0)
        length = 0.12 * scale
        from tigre.utilities.visualization.geometry_markers import draw_world_axes
        self._origin_arrows = draw_world_axes(self._axes, length)
        self._cor_artist = None
        self._cor_label = None
        self._cor_line = None
        cor = parameters.get("cor_mm", 0)
        if cor and parameters["source_motion"] == "Rotating source + detector":
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            longitude, latitude = np.meshgrid(
                np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
            radius = 0.012 * scale  # Display size only; sphere centre is the COR shift.
            sphere = radius * np.stack((np.sin(latitude) * np.cos(longitude),
                np.sin(latitude) * np.sin(longitude), np.cos(latitude)), axis=-1)
            self._cor_sphere_vertices = np.stack((sphere[:-1, :-1], sphere[:-1, 1:],
                sphere[1:, 1:], sphere[1:, :-1]), axis=2).reshape(-1, 4, 3)
            self._cor_artist = Poly3DCollection(
                self._cor_sphere_vertices, facecolors="darkorange", edgecolors="darkorange", linewidths=0,
                shade=True, label=f"COR shift: {cor:+g} mm (actual scale)")
            self._axes.add_collection3d(self._cor_artist)
            self._cor_label_offset = np.array([2 * radius, 2 * radius, 2 * radius])
            self._cor_label = self._axes.text(
                0, 0, 0, "COR", color="darkorange", fontweight="bold")
            self._cor_line, = self._axes.plot(
                [0, 0], [0, 0], [0, 0], color="darkorange", linewidth=2)
            self._update_cor_marker(parameters, 0)

    def _update_cor_marker(self, parameters, view_index):
        if getattr(self, "_cor_artist", None) is None:
            return
        count = len(self.surface.source_positions_mm)
        angle = np.deg2rad(parameters.get("orbit_start_deg", 0)) + np.linspace(
            0, np.deg2rad(parameters.get("scan_span_deg", 360)), count,
            endpoint=parameters.get("scan_span_deg", 360) < 360)[view_index]
        shift = parameters["cor_mm"] * np.array([-np.sin(angle), np.cos(angle), 0])
        self._cor_center = shift
        self._cor_artist.set_verts(self._cor_sphere_vertices + shift)
        label_position = shift + self._cor_label_offset
        self._cor_label.set_position_3d(label_position)
        self._cor_line.set_data_3d([0, shift[0]], [0, shift[1]], [0, shift[2]])

    def _panel_vertices(self, parameters, view_index):
        n_rows, n_columns = parameters["module_shape"]
        pitch_row, pitch_column = parameters["pixel_spacing_mm"]
        source = self.surface.source_positions_mm[view_index]
        if parameters["source_motion"] == "Fixed detector, moving source":
            orientation_source = self.surface.source_positions_mm[0]
        else:
            orientation_source = source
        if parameters["source_motion"] == "Fixed detector, arc source":
            gantry_angle = np.deg2rad(
                parameters["orbit_start_deg"] + parameters["scan_span_deg"] / 2.0)
        else:
            gantry_angle = np.arctan2(orientation_source[1], orientation_source[0])
        if parameters["layout"] == "Cylindrical":
            if self._panel_count == 1:
                panel_angles = np.zeros(1)
            else:
                panel_angles = np.linspace(
                    np.deg2rad(parameters["angular_span_deg"]) / 2.0,
                    -np.deg2rad(parameters["angular_span_deg"]) / 2.0,
                    self._panel_count)
        else:
            panel_angles = np.deg2rad(parameters["panel_angles_deg"])

        x_angles = _panel_tilts(parameters.get("panel_x_angles_deg", [0]), self._panel_count, "Panel x angles")
        y_angles = _panel_tilts(parameters.get("panel_y_angles_deg", [0]), self._panel_count, "Panel y angles")
        if parameters.get("cor_mm", 0) and parameters["source_motion"] == "Rotating source + detector":
            gantry_angle = np.deg2rad(parameters.get("orbit_start_deg", 0)) + np.linspace(
                0, np.deg2rad(parameters.get("scan_span_deg", 360)), len(self.surface.source_positions_mm),
                endpoint=parameters.get("scan_span_deg", 360) < 360)[view_index]
        panels = []
        for index, panel_angle in enumerate(panel_angles):
            start = index * (n_columns + parameters["column_gap"])
            grid = self.surface.pixel_centers_mm[
                view_index, :, start:start + n_columns, :]
            center = grid.mean(axis=(0, 1))
            angle = gantry_angle + panel_angle
            rotation = _panel_rotation(angle, x_angles[index], y_angles[index])
            column_axis = rotation[:, 1]
            row_axis = rotation[:, 2]
            half_width = n_columns * pitch_column / 2.0
            half_height = n_rows * pitch_row / 2.0
            panels.append([
                center - half_width * column_axis - half_height * row_axis,
                center + half_width * column_axis - half_height * row_axis,
                center + half_width * column_axis + half_height * row_axis,
                center - half_width * column_axis + half_height * row_axis,
            ])
        return panels

    def _render_frame(self, view_index):
        if self.surface is None or self._panel_collection is None:
            return
        view_index = int(np.clip(view_index, 0, len(self.surface.source_positions_mm) - 1))
        if getattr(self, "_surface_renderer", None) is not None:
            self._surface_renderer.update(view_index)
        else:
            self._panel_collection.set_verts(self._panel_vertices(self._parameters, view_index))
            source = self.surface.source_positions_mm[view_index]
            self._source_artist._offsets3d = ([source[0]], [source[1]], [source[2]])
        if not getattr(self, "_inspection_mode", False):
            self._update_cor_marker(self._parameters, view_index)
        self._frame_index = view_index
        self._canvas.draw_idle()

    def _on_frame_slider(self, value):
        if self.surface is not None:
            self._render_frame(int(float(value)))

    def toggle_animation(self):
        if self.surface is None or len(self.surface.source_positions_mm) < 2:
            return
        if self._animation_playing:
            self._stop_animation()
        else:
            self._animation_playing = True
            self._play_button.configure(text="Pause")
            self._schedule_next_frame()

    def _schedule_next_frame(self):
        if not self._animation_playing or self.surface is None:
            return
        n_views = len(self.surface.source_positions_mm)
        next_frame = (self._frame_index + 1) % n_views
        self._frame_slider.set(next_frame)
        self._animation_after = self.root.after(100, self._schedule_next_frame)

    def _stop_animation(self):
        self._animation_playing = False
        if getattr(self, "_play_button", None) is not None:
            self._play_button.configure(text="Play")
        if self._animation_after is not None:
            self.root.after_cancel(self._animation_after)
            self._animation_after = None

    def save_surface(self):
        from tkinter import messagebox

        if self.surface is None:
            messagebox.showinfo("No surface", "Preview a valid layout before saving.",
                                parent=self.root)
            return
        if self._compatibility_check_pending:
            return
        self._compatibility_check_pending = True
        self._save_button.configure(state=self.tk.DISABLED)
        self._status.set("Checking TIGRE compatibility...")
        surface = self.surface
        self._compatibility_surface = surface
        worker = threading.Thread(
            target=self._run_compatibility_check,
            args=(surface,),
            daemon=True,
        )
        try:
            worker.start()
        except RuntimeError as exc:
            self._compatibility_check_pending = False
            self._compatibility_surface = None
            self._save_button.configure(state=self.tk.NORMAL)
            self._status.set("Could not start the compatibility check.")
            messagebox.showerror(
                "Could not check compatibility", str(exc), parent=self.root)
            return
        self.root.after(50, self._poll_compatibility_result)

    def _run_compatibility_check(self, surface):
        try:
            result = check_tigre_compatibility(surface)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self._compatibility_queue.put(result)

    def _poll_compatibility_result(self):
        from tkinter import filedialog, messagebox

        try:
            result = self._compatibility_queue.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_compatibility_result)
            return
        self._compatibility_check_pending = False
        self._save_button.configure(state=self.tk.NORMAL)

        if self.surface is not self._compatibility_surface:
            self._compatibility_surface = None
            self._status.set("Geometry changed during the check; check again before saving.")
            messagebox.showwarning(
                "Geometry changed",
                "The preview changed while compatibility was being checked. "
                "Press Save surface again to check the current geometry.",
                parent=self.root,
            )
            return
        self._compatibility_surface = None

        if "error" in result:
            self._status.set("Compatibility check failed.")
            messagebox.showerror(
                "TIGRE compatibility check failed", result["error"], parent=self.root)
            return

        summary = (
            f"Iterative surface reconstruction: {result['iterative_message']}\n\n"
            f"fdk_surface: {result['fdk_message']}\n\n"
            "Continue to save this surface?"
        )
        self._status.set(
            "Compatibility checked: iterative methods supported; "
            + ("FDK supported." if result["fdk_supported"] else "FDK not supported."))
        if not messagebox.askyesno(
                "TIGRE compatibility", summary, parent=self.root):
            return

        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save detector surface",
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if not filename:
            return
        try:
            np.savez_compressed(
                filename,
                **surface_archive_arrays(self.surface,
                    self._inspection_geometry if getattr(self, "_inspection_mode", False) else self._builder_geometry),
            )
        except OSError as exc:
            messagebox.showerror("Could not save surface", str(exc), parent=self.root)
            return
        messagebox.showinfo("Surface saved", f"Saved detector surface to:\n{filename}",
                            parent=self.root)

    def remap_for_fdk(self):
        """Load measured projections and save them on an FDK-compatible surface."""
        from tkinter import filedialog, messagebox

        if self.surface is None:
            messagebox.showinfo(
                "No surface", "Preview a valid layout before remapping.", parent=self.root)
            return
        if self._remap_pending:
            return
        source_surface = self.surface
        input_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select projection data to remap",
            filetypes=(
                ("Projection arrays", "*.npy *.npz"),
                ("Raw float32 projections", "*.raw *.bin"),
                ("All files", "*.*"),
            ),
        )
        if not input_path:
            return
        if not messagebox.askyesno(
                "Warning: projection interpolation",
                "This creates new projection samples by linear interpolation onto "
                "an idealized circular, equiangular detector. Interpolation changes "
                "the measured data and may blur or bias the reconstruction. Detector "
                "gaps remain masked; unsupported layouts are rejected. The source "
                "must already follow a circular orbit. Continue?",
                parent=self.root):
            return
        output_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save remapped projections and surface",
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if not output_path:
            return
        self._remap_pending = True
        self._remap_surface = source_surface
        self._remap_button.configure(state=self.tk.DISABLED)
        self._status.set("Remapping projections for fdk_surface...")
        worker = threading.Thread(
            target=self._run_projection_remap,
            args=(input_path, output_path, source_surface),
            daemon=True,
        )
        try:
            worker.start()
        except RuntimeError as exc:
            self._remap_pending = False
            self._remap_button.configure(state=self.tk.NORMAL)
            messagebox.showerror(
                "Could not start remapping", str(exc), parent=self.root)
            return
        self.root.after(100, self._poll_projection_remap)

    def _run_projection_remap(self, input_path, output_path, surface):
        try:
            projections = _load_gui_projections(input_path, surface.valid_mask.shape)
            from tigre.utilities.detector_geometry_remap import (
                remap_projections_for_fdk,
            )

            remapped, target, report = remap_projections_for_fdk(projections, surface)
            np.savez_compressed(
                output_path,
                projections=remapped,
                source_positions_mm=target.source_positions_mm,
                pixel_centers_mm=target.pixel_centers_mm,
                valid_mask=target.valid_mask,
                remap_warning=report["warning"],
                remap_input_shape=np.asarray(report["input_shape"], dtype=np.int64),
                remap_target_radius_mm=report["target_radius_mm"],
                fdk_short_scan_supported=report["short_scan_supported"],
                fdk_short_scan_deficiency_deg=report["short_scan_deficiency_deg"],
            )
            result = {"path": output_path, "report": report}
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self._remap_queue.put(result)

    def _poll_projection_remap(self):
        from tkinter import messagebox

        try:
            result = self._remap_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_projection_remap)
            return
        self._remap_pending = False
        self._remap_button.configure(state=(
            self.tk.NORMAL if self.surface is not None else self.tk.DISABLED))
        geometry_changed = self.surface is not self._remap_surface
        self._remap_surface = None
        if "error" in result:
            self._status.set("Projection remapping failed.")
            messagebox.showerror(
                "Could not remap projections", result["error"], parent=self.root)
            return
        report = result["report"]
        coverage_warning = (
            "\n\nThe scan is short of FDK's recommended fan-angle coverage by "
            f"{report['short_scan_deficiency_deg']:.3f} degrees. The default "
            "fdk_surface call may reject it; detector remapping cannot fix "
            "insufficient angular coverage."
            if not report["short_scan_supported"] else "")
        preview_warning = (
            "\n\nThe preview changed while remapping was running. The saved "
            "archive matches the geometry that was previewed when the job began."
            if geometry_changed else "")
        self._status.set(
            f"Remapped {report['input_shape']} to {report['output_shape']}; "
            f"target radius {report['target_radius_mm']:.3f} mm.")
        messagebox.showwarning(
            "Remapped data saved",
            f"{report['warning']}\n\n"
            f"Input shape: {report['input_shape']}\n"
            f"Output shape: {report['output_shape']}\n"
            f"Saved archive: {result['path']}\n\n"
            "Use its `projections` and surface arrays together with "
            "fdk_surface. Confirm FDK scan coverage before reconstructing."
            f"{coverage_warning}{preview_warning}",
            parent=self.root,
        )


def _load_gui_projections(filename, expected_shape):
    """Load .npy/.npz arrays or a raw float32 array matching the preview."""
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix in (".raw", ".bin"):
        values = np.fromfile(path, dtype="<f4")
    elif suffix in (".npy", ".npz"):
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            values = loaded
        else:
            try:
                if "projections" in loaded.files:
                    values = loaded["projections"]
                elif len(loaded.files) == 1:
                    values = loaded[loaded.files[0]]
                else:
                    raise ValueError(
                        "The .npz file must contain a 'projections' array or "
                        "exactly one array.")
            finally:
                loaded.close()
    else:
        raise ValueError("Projection input must be .npy, .npz, .raw, or .bin.")

    values = np.asarray(values)
    if values.shape == expected_shape[::2] and expected_shape[1] == 1:
        values = values[:, None, :]
    elif values.size == int(np.prod(expected_shape)) and suffix in (".raw", ".bin"):
        values = values.reshape(expected_shape)
    if values.shape != expected_shape:
        raise ValueError(
            f"Projection array must have shape {expected_shape}; got {values.shape}.")
    return np.asarray(values, dtype=np.float32)


def launch_detector_geometry_gui(geometry=None, angles=None, surface=None):
    """Open the builder, or inspect supplied TIGRE geometry/surface read-only."""
    import tkinter as tk

    root = tk.Tk()
    DetectorGeometryBuilder(root, geometry=geometry, angles=angles, surface=surface)
    root.mainloop()


if __name__ == "__main__":
    launch_detector_geometry_gui()
