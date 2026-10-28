"""Interactive desktop builder for modular detector surfaces."""

import sys
import queue
import threading
import time
from pathlib import Path

import numpy as np

MAX_GUI_VIEWS = 6000
MAX_MODULE_DIMENSION = 4096
MAX_MODULES = 256
MAX_TOTAL_PIXELS = 20_000_000


def acquisition_timing(n_views, acquisition):
    """One synchronized detector frame per source pulse, starting at t=0."""
    names = ("readout_period_ms", "firing_rate_hz", "rotation_period_s",
             "table_speed_mm_s", "playback_rate")
    if not isinstance(acquisition, dict) or set(acquisition) != set(names):
        raise ValueError("Acquisition requires readout period, firing rate, rotation period, table speed and playback rate.")
    values = {name: float(acquisition[name]) for name in names}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Acquisition parameters must be finite.")
    if any(values[name] <= 0 for name in names if name != "table_speed_mm_s"):
        raise ValueError("Timing periods, firing rate and playback rate must be positive.")
    if values["readout_period_ms"] * values["firing_rate_hz"] > 1000 * (1 + 1e-12):
        raise ValueError("Source firing rate exceeds detector capacity. Reduce firing rate or readout period; one frame is acquired per pulse.")
    if (isinstance(n_views, (bool, np.bool_)) or not isinstance(n_views, (int, np.integer))
            or not 1 <= n_views <= MAX_GUI_VIEWS):
        raise ValueError(f"Views must be between 1 and {MAX_GUI_VIEWS}.")
    times = np.arange(n_views, dtype=float) / values["firing_rate_hz"]
    return values, times


def _view_angles(parameters, count):
    acquisition = parameters.get("acquisition")
    if acquisition is not None:
        values, times = acquisition_timing(count, acquisition)
        return np.deg2rad(parameters.get("orbit_start_deg", 0)) + 2 * np.pi * times / values["rotation_period_s"]
    span = parameters.get("scan_span_deg", 360)
    return np.deg2rad(parameters.get("orbit_start_deg", 0)) + np.linspace(
        0, np.deg2rad(span), count, endpoint=span < 360)

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
        "Choose Moving with Source to rotate the detector with a cylindrical source orbit, "
        "or Fixed Position to keep it stationary. Fixed Position uses the selected "
        "Linear source trajectory or Cylindrical source arc. Moving with Source "
        "switches a Linear source to Cylindrical. Confirm applies changes. "
        "Confirm rejects same-side panels with no transmitting rays, sources/detectors "
        "inside the volume, and views with no ray crossing the volume. "
        "Use iterative reconstruction for limited-angle scans."
    ),
    "Layout": (
        "Choose how module centers are arranged. Cylindrical places panels along "
        "an arc; Piecewise linear lets you specify panel centers and angles, "
        "including a U-shaped detector."
    ),
    "Views": (
        "Number of acquired views (1 to 6000), shared with Movement. With timing, "
        "one view is acquired per pulse at t=i/firing_rate. Otherwise angular scans "
        "sample the scan span and fixed-linear scans sample the source start/end line."
    ),
    "Orbit start (degrees)": (
        "Starting source angle about +z for a rotating source and detector. "
        "Used in rotating and fixed-arc modes; ignored in fixed-linear mode."
    ),
    "Scan span (degrees)": (
        "Angular distance travelled by the rotating source and detector. A "
        "360-degree scan omits the duplicate endpoint; shorter scans include "
        "both endpoints. Applies to rotating and fixed-arc sources; ignored for "
        "fixed-linear sources. Timing mode derives the span instead and permits multiple turns."
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
        "coordinates. Used only in fixed-detector, linear-source mode."
    ),
    "Source end (x, y, z mm)": (
        "End point of the source's straight-line trajectory in world x, y, z "
        "coordinates. Used only in fixed-detector, linear-source mode."
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

DETECTOR_FIELD_HELP.update({
    "Source trajectory": "Cylindrical follows a circular orbit about world +z using radius, height, start angle and scan span. Linear samples the world x, y, z start/end line and requires a fixed detector. Selecting Cylindrical after Linear keeps the detector fixed for an arc source; use Detector motion to rotate it with the source. Confirm applies the selection.",
    "Voxel counts": "Three positive integer voxel counts in TIGRE z, y, x order. Voxel size equals full physical size divided by these counts; the preview allocates no volume array.",
    "Full size (mm)": "Full positive physical extent of the volume along z, y, x, in mm. These are full lengths, not half-widths. The box is centered at the center offset.",
    "Center offset (mm)": "Initial volume center offOrigin in z, y, x mm. Shared with Start position in Movement. Source coordinates use x, y, z instead.",
    "Start position (z, y, x mm)": "Initial volume center in z, y, x mm, shared with the Volume tab's center offset. Movement starts here at the first view (t=0).",
    "End position (z, y, x mm)": "Volume center at the last acquired view, in z, y, x mm. Endpoint mode interpolates all three coordinates and requires at least two views for different endpoints. In timed speed mode this is a calculated, read-only result.",
    "Move between start and end positions": "Enable to interpolate the volume center from Start to End over the acquired views. With timing enabled, z speed is derived from z travel divided by last-view time. Disable to use entered z speed with timing, or a stationary volume without timing.",
    "Use timing to derive orbit and z motion": "Enable synchronized acquisition: one frame per source pulse, with t=i/firing_rate. Requires rotating source + detector. Rotation period determines angle, and z speed or endpoints determine volume motion. Manual scan span is ignored.",
    "Detector readout period (ms)": "Minimum time between detector frames, in milliseconds. Firing rate must be at most 1000/readout_period. This model acquires one frame per pulse; it does not discard pulses or combine exposures.",
    "Source firing rate (Hz)": "Source pulses per second and acquired views per second. View i occurs at i/rate seconds. For N views, the last view is at (N-1)/rate; the acquisition window is N/rate. Rate must fit detector readout capacity.",
    "Rotation period (s/turn)": "Positive time for one full source/detector turn. Angle at time t is orbit_start + 360*t/period. Views per turn equal firing_rate*period; timed scans may include multiple turns.",
    "Volume z speed (mm/s)": "Signed volume speed along world z. Positive moves toward +z; x/y stay fixed in speed mode. Endpoint mode derives this value from (end_z-start_z)/last_view_time. Travel per turn is z_speed*rotation_period.",
    "Playback speed (x real time)": "Positive animation speed relative to acquisition time (1 is real time, 0.5 is half speed). Display frames may be skipped to keep pace; acquisition times and saved views do not change.",
    "Views / source pulses": "Shared with Source Views. Number of detector frames/pulses, not number of rotations. With timing, last-view time=(N-1)/firing_rate. More views increase duration, rotation count and z travel in speed mode.",
})

TAB_HELP = {
    "Source": "Set source trajectory and view sampling. Source positions use world x, y, z mm; angles are degrees about +z. Fixed-linear mode uses start/end coordinates; circular modes use radius, height and scan span. Hover or focus a field for details, then Confirm to apply.",
    "Detectors": "Choose panel layout, pixel counts, pitches, panel alignment and detector motion. Cylindrical fields define an arc; piecewise-linear lists define individual panels. List... edits per-panel values. Confirm applies changes; masked gap columns are not measured samples.",
    "Volume": "Set voxel counts, full physical size and initial center in TIGRE z, y, x order. Voxel size is size/counts. Center offset is shared with Movement Start position. The common 3D preview shows the volume box at its per-view center.",
    "Movement": "Configure timing and volume motion. Timing mode requires rotating hardware and one frame per pulse. Choose z-speed mode or start/end interpolation; endpoints use z, y, x mm. Play and the slider animate the volume in the same world-frame scene. Moving offOrigin is for preview; surface reconstruction requires a fixed object coordinate frame.",
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
        panel_y_angles_deg=(0.0,), cor_mm=0.0, acquisition=None):
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
    if acquisition is not None:
        if source_motion != "Rotating source + detector":
            raise ValueError("Timed helical acquisition requires rotating source + detector.")
        acquisition_timing(n_views, acquisition)
    if source_motion in ("Rotating source + detector", "Fixed detector, arc source"):
        if not np.isfinite(source_radius_mm) or source_radius_mm <= 0:
            raise ValueError("Source radius must be a positive finite number.")
        if not np.isfinite(source_z_mm):
            raise ValueError("Source height must be finite.")
        if not np.isfinite(orbit_start_deg):
            raise ValueError("Orbit start angle must be finite.")
        if (not np.isfinite(scan_span_deg) or scan_span_deg <= 0
                or (acquisition is None and scan_span_deg > 360)):
            raise ValueError("Scan span must be positive and at most 360 degrees unless acquisition timing is enabled.")
        source_start = np.array(
            [source_radius_mm, 0.0, source_z_mm], dtype=np.float64)
        angles = _view_angles(dict(orbit_start_deg=orbit_start_deg,
            scan_span_deg=scan_span_deg, acquisition=acquisition), int(n_views))
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
        from tkinter import font as tkfont

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.tk = tk
        self.ttk = ttk
        self.root = root if root is not None else tk.Tk()
        self.root.title("TIGRE CT Simulation Builder")
        self._tab_font = tkfont.nametofont("TkDefaultFont", root=self.root).copy()
        tab_size = self._tab_font.cget("size")
        self._tab_font.configure(size=tab_size + 2 if tab_size > 0 else tab_size - 3, weight="bold")
        ttk.Style(self.root).configure("DetectorBuilder.TNotebook.Tab",
                                      font=self._tab_font, padding=(12, 7))
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1400, screen_width - 80)}x{min(900, screen_height - 100)}")
        self.root.minsize(min(960, screen_width - 80), min(600, screen_height - 100))
        self.surface = None
        self._inspection_mode = False
        self._inspection_geometry = None
        self._builder_geometry = None
        self._surface_renderer = None
        self._fields = {}
        self._entries = {}
        self._animation_after = None
        self._animation_playing = False
        self._confirmed_once = False
        self._last_saved_yaml = None
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
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        controls.rowconfigure(0, weight=1)
        controls.columnconfigure(0, weight=1)
        preview = ttk.Frame(outer)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)

        self._settings_tabs = ttk.Notebook(controls, style="DetectorBuilder.TNotebook")
        self._settings_tabs.grid(row=0, column=0, columnspan=2, sticky="nsew")
        footer = ttk.Frame(outer)
        footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(0, weight=1)
        action_controls = ttk.Frame(controls)
        action_controls.grid(row=3, column=0, columnspan=2, sticky="ew")
        playback_controls = ttk.Frame(footer)
        playback_controls.grid(row=0, column=0, sticky="ew")
        playback_controls.columnconfigure(0, weight=1)
        self._settings_scrolls = []

        def settings_page(title, notebook=None):
            notebook = self._settings_tabs if notebook is None else notebook
            page = ttk.Frame(notebook)
            notebook.add(page, text=title)
            viewport = tk.Canvas(page, width=380, height=400, highlightthickness=0)
            scrollbar = ttk.Scrollbar(page, orient=tk.VERTICAL, command=viewport.yview)
            viewport.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            viewport.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            content = ttk.Frame(viewport, padding=6)
            window = viewport.create_window(0, 0, anchor="nw", window=content)
            content.bind("<Configure>", lambda event: viewport.configure(scrollregion=viewport.bbox("all")))
            viewport.bind("<Configure>", lambda event: viewport.itemconfigure(window, width=event.width))
            self._settings_scrolls.append((viewport, content))
            return content

        source_controls = settings_page("Source")
        detector_controls = settings_page("Detectors")
        timing_controls = settings_page("Movement")
        source_controls.columnconfigure(1, weight=1)
        detector_controls.columnconfigure(1, weight=1)

        volume_controls = settings_page("Volume")
        self._settings_tabs.insert("end", timing_controls.master.master)
        ttk.Label(volume_controls, text="Reconstruction volume: z, y, x order").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=4)
        self._volume_fields = {}
        for column, (name, label, default) in enumerate((
                ("nVoxel", "Voxel counts", "128, 128, 128"),
                ("sVoxel", "Full size (mm)", "256, 256, 256"),
                ("offOrigin", "Center offset (mm)", "0, 0, 0"))):
            self._add_help_label(volume_controls, label).grid(row=column + 1, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=default)
            self._volume_fields[name] = variable
            entry = ttk.Entry(volume_controls, textvariable=variable, width=18)
            entry.grid(row=column + 1, column=1, sticky="ew", pady=4)
            self._entries[name] = entry
            self._bind_parameter_help(entry, label)
        volume_controls.columnconfigure(1, weight=1)
        self._voxel_spacing = tk.StringVar(value="Select Confirm to apply volume settings.")
        ttk.Label(volume_controls, textvariable=self._voxel_spacing).grid(
            row=4, column=0, columnspan=2, sticky="w")
        self._volume_motion = tk.BooleanVar(value=False)
        self._volume_motion_toggle = ttk.Checkbutton(timing_controls,
            text="Move between start and end positions",
            variable=self._volume_motion, command=self._update_field_states)
        self._volume_motion_toggle.grid(row=8, column=0, columnspan=2, sticky="w", pady=8)
        self._bind_parameter_help(self._volume_motion_toggle, "Move between start and end positions")
        self._add_help_label(timing_controls, "Start position (z, y, x mm)").grid(row=9, column=0, sticky="w", pady=4)
        self._entries["volume_start"] = ttk.Entry(timing_controls,
            textvariable=self._volume_fields["offOrigin"], width=18)
        self._entries["volume_start"].grid(row=9, column=1, sticky="ew")
        self._bind_parameter_help(self._entries["volume_start"], "Start position (z, y, x mm)")
        self._volume_end = tk.StringVar(value="0, 0, 0")
        self._entries["volume_end"] = ttk.Entry(timing_controls,
            textvariable=self._volume_end, width=18)
        self._add_help_label(timing_controls, "End position (z, y, x mm)").grid(row=10, column=0, sticky="w", pady=4)
        self._entries["volume_end"].grid(row=10, column=1, sticky="ew")
        self._bind_parameter_help(self._entries["volume_end"], "End position (z, y, x mm)")
        ttk.Label(timing_controls, text="Start is shared with the structural volume center. "
            "With timing enabled, start/end positions determine speed. In timed speed mode, z speed determines the end.",
            wraplength=350).grid(row=11, column=0, columnspan=2, sticky="w", pady=8)

        self._timing_enabled = tk.BooleanVar(value=False)
        self._timing_toggle = ttk.Checkbutton(timing_controls,
            text="Use timing to derive orbit and z motion",
            variable=self._timing_enabled, command=self._update_field_states)
        self._timing_toggle.grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        self._bind_parameter_help(self._timing_toggle, "Use timing to derive orbit and z motion")
        self._timing_fields = {}
        for index, (name, label, default) in enumerate((
                ("readout_period_ms", "Detector readout period (ms)", "10"),
                ("firing_rate_hz", "Source firing rate (Hz)", "100"),
                ("rotation_period_s", "Rotation period (s/turn)", "1"),
                ("table_speed_mm_s", "Volume z speed (mm/s)", "10"),
                ("playback_rate", "Playback speed (x real time)", "1"))):
            row = index + 1
            self._add_help_label(timing_controls, label).grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(value=default)
            self._timing_fields[name] = variable
            entry = ttk.Entry(timing_controls, textvariable=variable, width=18)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            self._entries[name] = entry
            self._bind_parameter_help(entry, label)
        timing_controls.columnconfigure(1, weight=1)
        self._timing_summary = tk.StringVar(value="One frame per pulse. Positive z speed moves the volume along world +z.")
        ttk.Label(timing_controls, textvariable=self._timing_summary,
            wraplength=350).grid(row=6, column=0, columnspan=2, sticky="w", pady=8)

        self._detector_stationary = False
        self._source_trajectory = tk.StringVar(value="Cylindrical")
        self._add_help_label(source_controls, "Source trajectory").grid(row=0, column=0, sticky="w", pady=2)
        self._source_trajectory_box = ttk.Combobox(source_controls,
            textvariable=self._source_trajectory, values=("Cylindrical", "Linear"),
            state="readonly", width=16)
        self._source_trajectory_box.grid(row=0, column=1, sticky="ew", pady=2)
        self._source_trajectory_box.bind("<<ComboboxSelected>>", self._on_source_trajectory_changed)
        self._bind_parameter_help(self._source_trajectory_box, "Source trajectory")
        self._add_help_label(detector_controls, "Detector motion").grid(
            row=0, column=0, sticky="w", pady=2)
        self._detector_motion = tk.StringVar(value="Moving with Source")
        self._detector_motion_box = ttk.Combobox(detector_controls,
            textvariable=self._detector_motion, values=("Moving with Source", "Fixed Position"),
            state="readonly", width=18)
        self._detector_motion_button = self._detector_motion_box  # Retain the existing widget alias.
        self._detector_motion_box.bind("<<ComboboxSelected>>", self._on_detector_motion_changed)
        self._detector_motion_button.grid(row=0, column=1, sticky="ew", pady=2)
        self._bind_parameter_help(self._detector_motion_button, "Detector motion")

        self._layout = tk.StringVar(value="Cylindrical")
        self._add_help_label(detector_controls, "Layout").grid(
            row=1, column=0, sticky="w", pady=2)
        layout_box = ttk.Combobox(
            detector_controls, textvariable=self._layout,
            values=("Cylindrical", "Piecewise linear"), state="readonly", width=20)
        self._layout_box = layout_box
        layout_box.grid(row=1, column=1, sticky="ew", pady=2)
        layout_box.bind("<<ComboboxSelected>>", self._update_module_count_field)
        self._bind_parameter_help(layout_box, "Layout")

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
        source_row, detector_row = 1, 2
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
            entry = ttk.Entry(parent, textvariable=variable, width=16)
            self._entries[name] = entry
            self._bind_parameter_help(entry, label)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            if name in self._panel_list_fields:
                button = ttk.Button(parent, text="List...", width=6,
                    command=lambda field=name, title=label: self._show_panel_list(field, title))
                button.grid(row=row, column=2, padx=(4, 0), pady=2)
                self._list_buttons[name] = button
                self._bind_parameter_help(button, label)
            if name == "n_modules":
                self._module_count_label = label_widget
                self._module_count_entry = entry
        self._default_fields = {name: default for _, name, default in fields}
        self._add_help_label(timing_controls, "Views / source pulses").grid(row=7, column=0, sticky="w", pady=4)
        self._entries["movement_views"] = ttk.Entry(timing_controls,
            textvariable=self._fields["n_views"], width=18)
        self._entries["movement_views"].grid(row=7, column=1, sticky="ew")
        self._bind_parameter_help(self._entries["movement_views"], "Views / source pulses")
        self._update_module_count_field()
        self._fields["transverse_centers_mm"].trace_add(
            "write", lambda *_: self._update_module_count_field())

        help_frame = ttk.LabelFrame(controls, text="Parameter help", padding=4)
        self._help_frame = help_frame
        help_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self._help_text = tk.Text(
            help_frame, height=3, width=34, wrap=tk.WORD, state=tk.DISABLED)
        help_scroll = ttk.Scrollbar(help_frame, orient=tk.VERTICAL, command=self._help_text.yview)
        self._help_text.configure(yscrollcommand=help_scroll.set)
        help_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._settings_tabs.bind("<<NotebookTabChanged>>", self._on_help_tab_changed)
        self._on_help_tab_changed()
        checks = ttk.LabelFrame(controls, text="Confirmation checks", padding=4)
        checks.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._validation_text = tk.Text(checks, height=4, width=34, wrap=tk.WORD, state=tk.DISABLED)
        checks_scroll = ttk.Scrollbar(checks, orient=tk.VERTICAL, command=self._validation_text.yview)
        self._validation_text.configure(yscrollcommand=checks_scroll.set)
        checks_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._validation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._set_validation_results("Not confirmed. Press Confirm to see physical and reconstruction checks.")

        buttons = ttk.Frame(action_controls)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self._preview_button = ttk.Button(buttons, text="Confirm", command=self.confirm_parameters)
        self._preview_button.pack(side=tk.LEFT, padx=(0, 4))
        self._save_button = ttk.Button(
            buttons, text="Save surface...", command=self.save_surface)
        self._save_button.pack(side=tk.LEFT)
        self._remap_button = ttk.Button(
            buttons, text="Remap for FDK...", command=self.remap_for_fdk)
        self._remap_button.pack(side=tk.LEFT, padx=(4, 0))
        self._remap_button.configure(state=tk.DISABLED)
        loading = ttk.Frame(action_controls)
        loading.grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Button(loading, text="Load geometry...", command=self.load_geometry).pack(side=tk.LEFT)
        self._builder_button = ttk.Button(loading, text="Return to builder", command=self.return_to_builder, state=tk.DISABLED)
        self._builder_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(loading, text="Save YAML...", command=self.save_geometry_settings).pack(side=tk.LEFT)
        self._export_recon_button = ttk.Button(action_controls, text="Export recon script...",
            command=self.export_reconstruction_script, state=tk.DISABLED)
        self._export_recon_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._status = tk.StringVar(value="Set parameters, then select Confirm.")
        ttk.Label(
            action_controls, textvariable=self._status, wraplength=350, justify=tk.LEFT
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=4)

        self._figure = Figure(figsize=(8, 7), dpi=100)
        self._axes = self._figure.add_subplot(111, projection="3d")
        main_plot = preview
        self._motion_summary = tk.StringVar(value="Select Confirm to display scan motion.")
        ttk.Label(timing_controls, textvariable=self._motion_summary, wraplength=350).grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=4)
        self._canvas = FigureCanvasTkAgg(self._figure, master=main_plot)
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._motion_current = tk.StringVar(value="")
        ttk.Label(timing_controls, textvariable=self._motion_current, wraplength=350).grid(
            row=13, column=0, columnspan=2, sticky="ew", pady=4)
        self._motion_positions = None
        playback = ttk.Frame(playback_controls)
        playback.grid(row=1, column=0, sticky="ew")
        self._play_button = ttk.Button(
            playback, text="Play", command=self.toggle_animation)
        self._play_button.pack(side=tk.LEFT, padx=(0, 8))
        self._frame_slider = tk.Scale(
            playback, from_=0, to=0, orient=tk.HORIZONTAL,
            showvalue=True, label="View", command=self._on_frame_slider)
        self._frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._play_button.configure(state=tk.DISABLED)
        self._frame_slider.configure(state=tk.DISABLED)
        examples = ttk.Frame(playback_controls)
        examples.grid(row=2, column=0, sticky="ew")
        examples.columnconfigure(1, weight=1)
        ttk.Label(examples, text="Example:").grid(row=0, column=0)
        self._example = tk.StringVar(value=next(iter(GUI_EXAMPLES)))
        ttk.Combobox(examples, textvariable=self._example,
                     values=tuple(GUI_EXAMPLES), state="readonly", width=20).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(examples, text="Load example", command=self.load_example).grid(row=0, column=2)
        self._example_description = tk.StringVar(value="")
        description = ttk.Label(examples, textvariable=self._example_description, wraplength=420)
        description.grid(row=1, column=0, columnspan=3, sticky="ew")
        examples.bind("<Configure>", lambda event: description.configure(wraplength=max(100, event.width - 12)))
        for viewport, content in self._settings_scrolls:
            def bind_wheel(widget, canvas=viewport):
                def scroll(event):
                    if canvas.bbox("all")[3] > canvas.winfo_height():
                        canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                        return "break"
                widget.bind("<MouseWheel>", scroll, add="+")
                for child in widget.winfo_children():
                    bind_wheel(child, canvas)
            bind_wheel(viewport)
        if geometry is not None or surface is not None:
            self.inspect_geometry(geometry, angles=angles, surface=surface)
        else:
            self.update_preview()

    def load_example(self):
        """Reset the controls and preview the selected demonstration."""
        self._inspection_mode = False
        self._builder_geometry = None
        self._builder_button.configure(state=self.tk.DISABLED)
        self._timing_enabled.set(False)
        self._set_volume_controls(None)
        layout, stationary, overrides, description = GUI_EXAMPLES[self._example.get()]
        self._stop_animation()
        for name, value in self._default_fields.items():
            self._fields[name].set(overrides.get(name, value))
        self._layout.set(layout)
        self._detector_stationary = stationary
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
        offsets = np.atleast_2d(getattr(document.geometry, "offOrigin", np.zeros(3)))
        nonlinear = len(offsets) > 1 and not np.allclose(offsets, np.linspace(offsets[0], offsets[-1], len(offsets)))
        if document.settings is None or nonlinear:
            self.inspect_geometry(document.geometry, surface=document.surface)
            return
        settings = document.settings
        self._stop_animation()
        self._inspection_mode = False
        self._builder_geometry = document.geometry
        acquisition = settings.get("acquisition")
        self._timing_enabled.set(acquisition is not None)
        if acquisition is not None:
            for name, value in acquisition.items():
                self._timing_fields[name].set(str(value))
        self._set_volume_controls(document.geometry)
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
            volume = self._inspection_geometry if inspection else self._read_volume()
            if not inspection:
                from tigre.utilities.ct_physics import validate_transmission_geometry
                validate_transmission_geometry(build_modular_surface(**settings), volume)
            if inspection and self.surface is None:
                raise ValueError("No geometry to save.")
            path = filedialog.asksaveasfilename(parent=self.root,
                title="Save geometry YAML", defaultextension=".yaml",
                filetypes=(("YAML geometry", "*.yaml *.yml"),))
            if not path:
                return
            save_geometry_yaml(path, settings=settings,
                surface=self.surface if inspection else None,
                geometry=volume)
            self._last_saved_yaml = Path(path).resolve()
            self._export_recon_button.configure(state=self.tk.NORMAL)
            self._status.set(f"Saved geometry YAML: {path}")
        except (ValueError, TypeError, OSError, ImportError, MemoryError) as exc:
            messagebox.showerror("Cannot save geometry YAML", str(exc), parent=self.root)

    def export_reconstruction_script(self):
        """Export an example referring to the last successfully saved YAML."""
        from tkinter import filedialog, messagebox
        from tigre.utilities.geometry_recon_export import reconstruction_script_text
        if self._last_saved_yaml is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root,
            title=f"Export recon script for {self._last_saved_yaml.name}",
            initialfile=self._last_saved_yaml.stem + "_recon.py", defaultextension=".py",
            filetypes=(("Python script", "*.py"),))
        if not path:
            return
        try:
            text = reconstruction_script_text(self._last_saved_yaml, path)
            Path(path).write_text(text, encoding="utf-8")
            self._status.set(f"Exported recon script using {self._last_saved_yaml.name}: {path}")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Cannot export recon script", str(exc), parent=self.root)

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
        self._parameters = {"acquisition": getattr(geometry, "acquisition", None)}
        self._inspection_mode = True
        self._set_validation_results("Inspection mode: exact loaded geometry is displayed. Return to builder and Confirm to check edited settings; exported scripts validate before reconstruction.")
        self._inspection_geometry = geometry
        self._surface_renderer = SurfaceRenderer(self._axes, surface, geo=geometry)
        self._set_motion_details(geometry)
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
        label = self.ttk.Label(parent, text=text, cursor="question_arrow", wraplength=150)
        self._bind_parameter_help(label, text)
        return label

    def _bind_parameter_help(self, widget, name):
        for event in ("<Enter>", "<FocusIn>"):
            widget.bind(event, lambda _event, key=name: self._set_help(key), add="+")

    def _on_help_tab_changed(self, event=None):
        if not hasattr(self, "_help_text"):
            return
        tab = self._settings_tabs.tab(self._settings_tabs.select(), "text")
        self._help_frame.configure(text=f"Parameter help: {tab}")
        self._set_help(tab)

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
        if hasattr(self, "_source_trajectory_box"):
            self._source_trajectory.set("Linear" if self._detector_stationary is True else "Cylindrical")
            self._source_trajectory_box.configure(state="disabled" if inspection else "readonly")
        if hasattr(self, "_detector_motion_box"):
            self._detector_motion.set("Fixed Position" if self._detector_stationary else "Moving with Source")
            self._detector_motion_box.configure(state="disabled" if inspection else "readonly")
        for name in ("_volume_motion_toggle", "_timing_toggle"):
            if hasattr(self, name):
                getattr(self, name).configure(state=self.tk.DISABLED if inspection else self.tk.NORMAL)
        if hasattr(self, "_layout_box"):
            self._layout_box.configure(state="disabled" if inspection else "readonly")
            if not hasattr(self, "_detector_motion_box"):
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
        timed = hasattr(self, "_timing_enabled") and self._timing_enabled.get()
        for name in getattr(self, "_timing_fields", {}):
            self._entries[name].configure(state=self.tk.NORMAL if timed else self.tk.DISABLED)
        if timed:
            self._entries["scan_span_deg"].configure(state=self.tk.DISABLED)
            endpoints = self._volume_motion.get()
            self._entries["volume_end"].configure(state=self.tk.NORMAL if endpoints else self.tk.DISABLED)
            self._entries["table_speed_mm_s"].configure(state=self.tk.DISABLED if endpoints else self.tk.NORMAL)


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

    def _on_source_trajectory_changed(self, event=None):
        """Use supported source/detector combinations; Confirm rebuilds the scene."""
        if self._source_trajectory.get() == "Linear":
            self._detector_stationary = True
        elif self._detector_stationary is True:
            self._detector_stationary = "arc"
        self._update_field_states()
        self._set_help("Source trajectory")

    def _on_detector_motion_changed(self, event=None):
        if self._detector_motion.get() == "Moving with Source":
            self._detector_stationary = False
        else:
            self._detector_stationary = True if self._source_trajectory.get() == "Linear" else "arc"
        self._update_field_states()
        self._set_help("Detector motion")

    def toggle_detector_motion(self):
        """Toggle between a stationary detector and one rotating with the source."""
        self._detector_stationary = {False: True, True: "arc", "arc": False}[self._detector_stationary]
        state = ("Stationary (arc source)" if self._detector_stationary == "arc" else
                 "Stationary" if self._detector_stationary else "Moving with source")
        if not hasattr(self, "_detector_motion_box"):
            self._detector_motion_button.configure(text=state)
        self._update_field_states()
        self._help_text.configure(state=self.tk.NORMAL)
        self._help_text.delete("1.0", self.tk.END)
        self._help_text.insert("1.0", DETECTOR_FIELD_HELP["Detector motion"])
        self._help_text.insert(self.tk.END, f"\n\nSelected: {state}")
        self._help_text.configure(state=self.tk.DISABLED)

    def _set_help(self, name):
        if not hasattr(self, "_help_text"):
            return
        text = DETECTOR_FIELD_HELP.get(name, TAB_HELP.get(name, "Hover or focus a parameter for help."))
        if getattr(self, "_inspection_mode", False):
            text += "\n\nInspection mode: these builder settings are inactive. Return to builder to edit."
        self._help_text.configure(state=self.tk.NORMAL)
        self._help_text.delete("1.0", self.tk.END)
        self._help_text.insert("1.0", text)
        self._help_text.yview_moveto(0)
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
        acquisition = self._read_acquisition()
        scan_span = float(values["scan_span_deg"]) if stationary != True and acquisition is None else 360.0
        if acquisition is not None:
            if stationary:
                raise ValueError("Timing mode requires rotating source + detector.")
            timing, times = acquisition_timing(n_views, acquisition)
            scan_span = 360 * n_views / timing["firing_rate_hz"] / timing["rotation_period_s"]
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
            "acquisition": acquisition,
        }

    def _read_acquisition(self):
        if not hasattr(self, "_timing_enabled") or not self._timing_enabled.get():
            return None
        values = {name: float(variable.get()) for name, variable in self._timing_fields.items()
                  if name != "table_speed_mm_s"}
        values["table_speed_mm_s"] = 0.0
        if hasattr(self, "_volume_motion") and self._volume_motion.get():
            count = int(self._parse_numbers(self._fields["n_views"].get(), "Views", integer=True))
            _, times = acquisition_timing(count, values)
            start, end = self._movement_endpoints()
            if count == 1:
                if not np.array_equal(start, end):
                    raise ValueError("At least two views are required to move between different positions.")
            else:
                values["table_speed_mm_s"] = float((end[0] - start[0]) / times[-1])
        else:
            values["table_speed_mm_s"] = float(self._timing_fields["table_speed_mm_s"].get())
        return values

    def _movement_endpoints(self):
        start = np.atleast_1d(self._parse_numbers(self._volume_fields["offOrigin"].get(), "Start position"))
        end = np.atleast_1d(self._parse_numbers(self._volume_end.get(), "End position"))
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("Start and end positions require three values in z, y, x order.")
        return start, end

    def _read_volume(self):
        """Read TIGRE volume metadata without allocating voxel data."""
        from tigre.utilities.geometry_io import _volume
        values = {}
        for name, variable in self._volume_fields.items():
            vector = np.atleast_1d(self._parse_numbers(variable.get(), name))
            if vector.shape != (3,):
                raise ValueError(f"{name} requires three values in z, y, x order.")
            values[name] = vector
        acquisition = self._read_acquisition()
        if acquisition is not None:
            count = int(self._parse_numbers(self._fields["n_views"].get(), "Views", integer=True))
            timing, times = acquisition_timing(count, acquisition)
            if hasattr(self, "_volume_motion") and self._volume_motion.get():
                start, end = self._movement_endpoints()
                offsets = np.linspace(start, end, count)
            else:
                offsets = np.tile(values["offOrigin"], (count, 1))
                offsets[:, 0] += timing["table_speed_mm_s"] * times
            values["offOrigin"] = offsets
        elif self._volume_motion.get():
            start, end = self._movement_endpoints()
            count = int(self._parse_numbers(self._fields["n_views"].get(), "Views", integer=True))
            if not 1 <= count <= MAX_GUI_VIEWS:
                raise ValueError(f"Views must be between 1 and {MAX_GUI_VIEWS}.")
            if count == 1 and not np.array_equal(start, end):
                raise ValueError("At least two views are required to move between different positions.")
            values["offOrigin"] = np.linspace(start, end, count)
        volume = _volume(values)
        volume.motion_mode = ("endpoints" if hasattr(self, "_volume_motion") and self._volume_motion.get() else
                              "speed" if acquisition is not None else "static")
        if acquisition is not None:
            volume.acquisition = timing
            volume.view_times_s = times
        return volume

    def _set_volume_controls(self, geometry):
        offsets = np.atleast_2d(getattr(geometry, "offOrigin", [0, 0, 0]))
        mode = getattr(geometry, "motion_mode", None)
        self._volume_motion.set(mode == "endpoints" if mode is not None else len(offsets) > 1)
        self._volume_end.set(", ".join(f"{v:g}" for v in offsets[-1]))
        for name, default in (("nVoxel", [128, 128, 128]),
                              ("sVoxel", [256, 256, 256]),
                              ("offOrigin", [0, 0, 0])):
            values = getattr(geometry, name, default)
            if name == "offOrigin":
                values = offsets[0]
            self._volume_fields[name].set(", ".join(f"{v:g}" for v in np.asarray(values).reshape(-1)))

    def confirm_parameters(self):
        """Explicitly apply the controls before allowing builder playback."""
        if self.update_preview():
            self._confirmed_once = True
            self._play_button.configure(state=self.tk.NORMAL)
            physical = self._physical_report
            lines = [f"Physical CT: PASS — {physical['views']} views; {physical['crossing_samples']:,} crossing samples."]
            offsets = np.atleast_2d(self._builder_geometry.offOrigin)
            moving = len(offsets) > 1 and not np.all(offsets == offsets[0])
            lines.append("CGLS geometry: supported" + (" after fixed-object conversion (exported script)." if moving else "."))
            if moving:
                lines.append("FDK: not assessed after volume-motion conversion; exported script checks the resulting trajectory.")
            else:
                try:
                    report = check_tigre_compatibility(self.surface)
                    lines.append("FDK geometry: " + ("PASS. Scan coverage still needs checking." if report["fdk_supported"]
                                                     else "NOT SUPPORTED. " + report["fdk_message"]))
                except Exception as exc:
                    lines.append(f"FDK check unavailable: {type(exc).__name__}: {exc}")
            lines.append("Geometry checks only; CUDA execution, sampling completeness and mechanical clearance are not certified.")
            self._set_validation_results("\n".join(lines))
            self._status.set("Confirmed. Physical CT checks passed; see Confirmation checks for algorithm details.")

    def _set_validation_results(self, text):
        if not hasattr(self, "_validation_text"):
            return
        self._validation_text.configure(state=self.tk.NORMAL)
        self._validation_text.delete("1.0", self.tk.END)
        self._validation_text.insert("1.0", text)
        self._validation_text.yview_moveto(0)
        self._validation_text.configure(state=self.tk.DISABLED)

    def update_preview(self):
        from tkinter import messagebox

        if getattr(self, "_inspection_mode", False):
            return

        try:
            parameters = self._read_parameters()
            volume = self._read_volume()
            surface = build_modular_surface(**parameters)
            from tigre.utilities.ct_physics import validate_transmission_geometry
            physical = validate_transmission_geometry(surface, volume)
        except (TypeError, ValueError, OverflowError, MemoryError) as exc:
            self._set_validation_results(f"Confirm: FAILED\n{exc}\nPrevious confirmed scene retained.")
            self._status.set("Confirm failed. See Confirmation checks; previous scene retained.")
            messagebox.showerror("Invalid detector layout", str(exc), parent=self.root)
            return False
        self._stop_animation()
        self._clear_preview()
        self.surface = surface
        self._builder_geometry = volume
        self._physical_report = physical
        self._parameters = parameters
        self._voxel_spacing.set("Voxel size (z, y, x mm): " +
            ", ".join(f"{v:.6g}" for v in volume.dVoxel))
        if parameters.get("acquisition") is not None:
            timing, times = acquisition_timing(parameters["n_views"], parameters["acquisition"])
            duration = parameters["n_views"] / timing["firing_rate_hz"]
            self._timing_fields["table_speed_mm_s"].set(f"{timing['table_speed_mm_s']:.12g}")
            self._volume_end.set(", ".join(f"{v:.12g}" for v in volume.offOrigin[-1]))
            self._timing_summary.set(
                f"Frame interval: {1000 / timing['firing_rate_hz']:.6g} ms; "
                f"acquisition window: {duration:.6g} s; last view: {times[-1]:.6g} s.\n"
                f"Turns in window: {duration / timing['rotation_period_s']:.6g} "
                f"({times[-1] / timing['rotation_period_s']:.6g} to last view); "
                f"views/turn: {timing['firing_rate_hz'] * timing['rotation_period_s']:.6g}; "
                f"z travel to last view: {timing['table_speed_mm_s'] * times[-1]:.6g} mm; "
                f"travel per turn: {timing['table_speed_mm_s'] * timing['rotation_period_s']:.6g} mm.")
        else:
            self._timing_summary.set("Timing disabled: scan span and volume end offset define the scan.")

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
        self._set_motion_details(volume)
        self._add_coordinate_markers(parameters)
        axes.legend(loc="upper right")
        self._frame_index = 0
        self._play_button.configure(state=self.tk.NORMAL if self._confirmed_once else self.tk.DISABLED)
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
        return True

    def _clear_preview(self):
        self._set_validation_results("Displayed geometry changed. Press Confirm to refresh check results.")
        if hasattr(self, "_motion_summary"):
            self._motion_summary.set("Select Confirm to display scan motion.")
            self._motion_current.set("")
        self._motion_positions = None
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

    def _set_motion_details(self, geometry):
        """Retain speed readouts while the common renderer shows world motion."""
        source = np.asarray(self.surface.source_positions_mm)
        offsets = np.atleast_2d(getattr(geometry, "offOrigin", np.zeros(3)))
        self._motion_positions = source - offsets[:, ::-1]
        self._motion_speeds = None
        acquisition = getattr(geometry, "acquisition", None)
        if acquisition is not None:
            timing, times = acquisition_timing(len(self.surface.source_positions_mm), acquisition)
            self._motion_summary.set(
                f"Firing: {timing['firing_rate_hz']:g} Hz; readout: {timing['readout_period_ms']:g} ms; "
                f"rotation: {timing['rotation_period_s']:g} s/turn.\n"
                f"Volume z speed: {timing['table_speed_mm_s']:g} mm/s; "
                f"travel/turn: {timing['table_speed_mm_s'] * timing['rotation_period_s']:g} mm; "
                f"turns to last view: {times[-1] / timing['rotation_period_s']:.6g}.")
            if len(times) > 1:
                source = np.asarray(self.surface.source_positions_mm)
                relative = self._motion_positions
                centers = source - relative
                self._motion_speeds = np.stack([
                    np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(times)
                    for points in (source, centers, relative)])
        else:
            self._motion_summary.set("Enable acquisition timing to display physical speeds and scan duration.")

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
        angle = _view_angles(parameters, count)[view_index]
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
            gantry_angle = _view_angles(parameters, len(self.surface.source_positions_mm))[view_index]
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
        if getattr(self, "_motion_positions", None) is not None:
            point = self._motion_positions[view_index]
            details = "Relative source (x, y, z mm): " + ", ".join(f"{v:.6g}" for v in point)
            if self._motion_speeds is not None:
                speeds = self._motion_speeds[:, max(0, view_index - 1)]
                details += (f"\nSample-to-sample speeds (mm/s): source {speeds[0]:.6g}; "
                            f"volume {speeds[1]:.6g}; relative {speeds[2]:.6g}.")
            self._motion_current.set(details)
        acquisition = (self._parameters or {}).get("acquisition")
        self._frame_slider.configure(label=(
            f"View (t = {view_index / acquisition['firing_rate_hz']:.6g} s)"
            if acquisition is not None else "View"))
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
            self._animation_started = time.monotonic()
            self._animation_start_frame = self._frame_index
            self._play_button.configure(text="Pause")
            self._schedule_next_frame()

    def _schedule_next_frame(self):
        if not self._animation_playing or self.surface is None:
            return
        n_views = len(self.surface.source_positions_mm)
        acquisition = (self._parameters or {}).get("acquisition")
        delay = 100
        if acquisition is not None:
            elapsed = (time.monotonic() - self._animation_started) * acquisition["playback_rate"]
            next_frame = (self._animation_start_frame + int(elapsed * acquisition["firing_rate_hz"])) % n_views
            delay = max(1, min(100, round(1000 / acquisition["firing_rate_hz"] / acquisition["playback_rate"])))
        else:
            next_frame = (self._frame_index + 1) % n_views
        self._frame_slider.set(next_frame)
        self._animation_after = self.root.after(delay, self._schedule_next_frame)

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
            messagebox.showinfo("No surface", "Confirm a valid layout before saving.",
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
            "Detector-surface compatibility (volume motion is not included):\n\n"
            f"Iterative surface reconstruction: {result['iterative_message']}\n\n"
            f"fdk_surface: {result['fdk_message']}\n\n"
            "Continue to save this surface?"
        )
        volume = self._inspection_geometry if getattr(self, "_inspection_mode", False) else self._builder_geometry
        offsets = np.atleast_2d(getattr(volume, "offOrigin", np.zeros(3)))
        if len(offsets) > 1 and not np.all(offsets == offsets[0]):
            summary = ("This volume moves between views. Use Export recon script to convert "
                       "source and detector coordinates to a fixed object frame for CGLS. "
                       "FDK may not support the resulting trajectory.\n\n" + summary)
        self._status.set(
            "Detector layout checked: iterative methods supported; "
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
                "No surface", "Confirm a valid layout before remapping.", parent=self.root)
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
            args=(input_path, output_path, source_surface,
                  self._inspection_geometry if self._inspection_mode else self._builder_geometry),
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

    def _run_projection_remap(self, input_path, output_path, surface, geometry=None):
        try:
            projections = _load_gui_projections(input_path, surface.valid_mask.shape)
            from tigre.utilities.detector_geometry_remap import (
                remap_projections_for_fdk,
            )

            remapped, target, report = remap_projections_for_fdk(projections, surface)
            np.savez_compressed(
                output_path,
                projections=remapped,
                **surface_archive_arrays(target, geometry),
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
