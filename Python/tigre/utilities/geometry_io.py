"""Versioned YAML settings and exact numeric surface archives.

YAML uses mm and degrees for modular builder settings. Explicit pixel arrays
remain in NPZ companions, loaded without pickle. No Tk window is required.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import inspect
import uuid
import numpy as np


def surface_archive_arrays(surface, geometry=None):
    """Numeric archive payload including optional module and volume metadata."""
    arrays = {name: getattr(surface, name) for name in (
        "source_positions_mm", "pixel_centers_mm", "valid_mask")}
    modules = getattr(surface, "_module_slices", None)
    if modules:
        arrays["module_slices"] = np.array([
            [r.start, r.stop, c.start, c.stop] for r, c in modules], dtype=np.int64)
    if geometry is not None:
        from tigre.utilities.visualization.surface_renderer import _volume_box
        origin, size = _volume_box(geometry)
        arrays["sVoxel"] = size[::-1]
        arrays["offOrigin"] = origin[::-1]
    return arrays


def load_geometry_archive(path):
    """Load explicit surfaces or standard TIGRE parameters from numeric NPZ."""
    from types import SimpleNamespace
    from tigre.utilities.detector_geometry import DetectorSurfaceGeometry, surface_from_geometry
    with np.load(path, allow_pickle=False) as archive:
        keys = {"source_positions_mm", "pixel_centers_mm", "valid_mask", "module_slices",
            "sVoxel", "offOrigin", "DSO", "DSD", "nDetector", "dDetector", "angles",
            "offDetector", "rotDetector", "COR", "nVoxel", "mode"}
        arrays = {key: archive[key] for key in archive.files if key in keys}
    if "source_positions_mm" in arrays:
        modules = arrays.get("module_slices")
        if modules is not None:
            if modules.ndim != 2 or modules.shape[1] != 4 or modules.dtype.kind not in "iu":
                raise ValueError("module_slices must be an integer (modules, 4) array.")
            modules = [(slice(int(a), int(b)), slice(int(c), int(d))) for a, b, c, d in modules]
        surface = DetectorSurfaceGeometry(arrays["source_positions_mm"],
            arrays["pixel_centers_mm"], arrays.get("valid_mask"), module_slices=modules)
        geometry = None
        if "sVoxel" in arrays:
            size = np.asarray(arrays["sVoxel"], dtype=float)
            offset = np.asarray(arrays.get("offOrigin", np.zeros(3)), dtype=float)
            if size.shape != (3,) or np.any(size <= 0) or not np.isfinite(size).all():
                raise ValueError("sVoxel must contain three positive finite values.")
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("offOrigin must contain three finite values.")
            geometry = SimpleNamespace(sVoxel=size, offOrigin=offset)
        return geometry, surface
    required = ("DSO", "DSD", "nDetector", "dDetector", "angles")
    if any(name not in arrays for name in required):
        raise ValueError("Archive needs source/pixel arrays, or DSO, DSD, nDetector, dDetector and angles.")
    if "mode" in arrays and arrays["mode"].item() != "cone":
        raise ValueError("Standard geometry inspection supports cone mode; supply an explicit surface otherwise.")
    import tigre
    geometry = tigre.geometry_default(high_resolution=False)
    for name in ("DSO", "DSD", "nDetector", "dDetector", "offDetector", "rotDetector", "COR",
                 "nVoxel", "sVoxel", "offOrigin"):
        if name in arrays:
            value = arrays[name]
            setattr(geometry, name, value.item() if value.ndim == 0 else value)
    geometry.sDetector = geometry.nDetector * geometry.dDetector
    geometry.dVoxel = geometry.sVoxel / geometry.nVoxel
    return geometry, surface_from_geometry(geometry, arrays["angles"])



@dataclass
class GeometryDocument:
    geometry: object
    surface: object
    settings: dict | None = None


def _yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML geometry files require PyYAML: install it in your TIGRE environment.") from exc
    return yaml


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _settings(values):
    from .detector_geometry_gui import build_modular_surface
    if not isinstance(values, dict):
        raise ValueError("builder must be a mapping of modular geometry settings.")
    try:
        bound = inspect.signature(build_modular_surface).bind(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid builder settings: {exc}") from exc
    bound.apply_defaults()
    return _plain(dict(bound.arguments))


def _volume(values):
    if values is None:
        return None
    if not isinstance(values, dict) or set(values) - {"sVoxel", "offOrigin"} or "sVoxel" not in values:
        raise ValueError("volume requires sVoxel and optional offOrigin, in TIGRE z/y/x order.")
    geo = SimpleNamespace(sVoxel=np.asarray(values["sVoxel"], dtype=float),
                          offOrigin=np.asarray(values.get("offOrigin", [0, 0, 0]), dtype=float))
    from .visualization.surface_renderer import _volume_box
    _volume_box(geo)
    return geo


def load_geometry_file(path):
    """Return geometry, exact/generated surface, and editable YAML settings.

    Version 1 YAML is either ``kind: modular`` with ``builder`` settings or
    ``kind: surface`` with a relative/local ``surface_file`` NPZ reference.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        geo, surface = load_geometry_archive(path)
        return GeometryDocument(geo, surface)
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Geometry files must be .yaml, .yml or .npz.")
    yaml = _yaml()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid geometry YAML: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != "tigre.geometry":
        raise ValueError("Expected a tigre.geometry YAML document.")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Unsupported geometry YAML version; expected 1.")
    if data.get("units") != {"length": "mm", "angle": "degree"}:
        raise ValueError("Geometry YAML units must be length: mm and angle: degree.")
    if data.get("rotation_convention") != "Rz(gantry+z) @ Ry(y) @ Rx(x)":
        raise ValueError("Unsupported rotation convention.")
    kind = data.get("kind")
    allowed = {"schema", "version", "units", "rotation_convention", "kind", "volume"}
    allowed.add("builder" if kind == "modular" else "surface_file")
    if set(data) - allowed:
        raise ValueError(f"Unknown YAML fields: {sorted(set(data) - allowed)}")
    geo = _volume(data.get("volume"))
    if kind == "modular":
        from .detector_geometry_gui import build_modular_surface
        settings = _settings(data.get("builder"))
        surface = build_modular_surface(**settings)
        return GeometryDocument(geo, surface, settings)
    if kind == "surface":
        reference = data.get("surface_file")
        if not isinstance(reference, str) or Path(reference).suffix.lower() != ".npz":
            raise ValueError("surface_file must reference a numeric .npz archive.")
        stored_geo, surface = load_geometry_archive(path.parent / reference)
        return GeometryDocument(geo if geo is not None else stored_geo, surface)
    raise ValueError("Geometry YAML kind must be modular or surface.")


def save_geometry_yaml(path, *, settings=None, surface=None, geometry=None):
    """Save editable settings or an exact surface with a unique NPZ companion.

    Builder settings are validated by constructing their surface. Inspection
    surfaces are never fitted to an approximate editable geometry.
    """
    path = Path(path)
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Geometry settings filename must end in .yaml or .yml.")
    yaml = _yaml()
    data = dict(schema="tigre.geometry", version=1,
                units={"length": "mm", "angle": "degree"},
                rotation_convention="Rz(gantry+z) @ Ry(y) @ Rx(x)")
    if geometry is not None:
        from .visualization.surface_renderer import _volume_box
        origin, size = _volume_box(geometry)
        data["volume"] = dict(sVoxel=size[::-1].tolist(), offOrigin=origin[::-1].tolist())
    if settings is not None:
        from .detector_geometry_gui import build_modular_surface
        values = _settings(settings)
        build_modular_surface(**values)
        data.update(kind="modular", builder=values)
    elif surface is not None:
        companion = path.with_name(path.stem + ".surface-" + uuid.uuid4().hex[:8] + ".npz")
        data.update(kind="surface", surface_file=companion.name)
        payload = surface_archive_arrays(surface, geometry)
        text = yaml.safe_dump(data, sort_keys=False)
        np.savez_compressed(companion, **payload)
        path.write_text(text, encoding="utf-8")
        return
    else:
        raise ValueError("Supply builder settings or an explicit detector surface.")
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
