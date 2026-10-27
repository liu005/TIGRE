"""CPU baseline for MATLAB's TrueBeam 2.0/2.7 XIM/XML loading workflow.

Includes acquisition geometry, acceleration/deceleration view selection,
blank/chamber normalization, anomaly clipping and optional horizontal median
filtering. DPS, scatter and beam-hardening models are not ported: requesting
them raises NotImplementedError. Defaults therefore differ from MATLAB.
"""
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import median_filter
from tigre.utilities.geometry import Geometry
from .xim import read_xim


def _element(parent, name):
    found = [item for item in parent if item.tag.split("}")[-1] == name]
    if len(found) != 1:
        raise ValueError(f"Expected one XML element {name}.")
    return found[0]


def _text(parent, name):
    value = _element(parent, name).text
    if value is None or not value.strip():
        raise ValueError(f"Empty XML value {name}.")
    return value.strip()


def _number(parent, name, positive=False):
    try:
        value = float(_text(parent, name))
    except ValueError as exc:
        raise ValueError(f"Invalid numeric XML value {name}.") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid numeric XML value {name}.")
    return value


def geometry_from_xml(datafolder, *, load_geometry=True):
    """Return Python geometry and parsed Scan.xml; convert MATLAB xyz to zyx."""
    folder = Path(datafolder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Varian acquisition folder does not exist: {folder}")
    path = folder / "Scan.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Expected Scan.xml in acquisition folder: {folder}")
    scan = ET.parse(path).getroot()
    if scan.tag.split("}")[-1] != "Scan":
        raise ValueError("Expected Scan XML root element.")
    acquisition = _element(scan, "Acquisitions")
    geo = Geometry()
    geo.mode = "cone"
    geo.DSD = _number(acquisition, "SID", True)
    geo.DSO = _number(acquisition, "SAD", True)
    counts = [_number(acquisition, name, True) for name in ("ImagerSizeY", "ImagerSizeX")]
    if any(value != int(value) for value in counts):
        raise ValueError("Detector pixel counts must be integers.")
    geo.nDetector = np.array(counts, dtype=int)
    if np.prod(geo.nDetector.astype(float)) > 64_000_000:
        raise ValueError("Detector dimensions exceed the XIM reader's pixel limit.")
    geo.dDetector = np.array([_number(acquisition, "ImagerResY", True),
                             _number(acquisition, "ImagerResX", True)])
    geo.sDetector = geo.nDetector*geo.dDetector
    offset = -_number(acquisition, "ImagerLat")
    geo.offDetector = np.array([0.0, offset])
    geo.offOrigin = np.zeros(3)
    geo.rotDetector = np.zeros(3)
    geo.COR, geo.accuracy = 0.0, .5
    recon_files = sorted(folder.rglob("Reconstruction.xml")) if load_geometry else []
    if len(recon_files) > 1:
        raise ValueError("Multiple Reconstruction.xml files; set load_geometry=False to estimate the volume.")
    if recon_files:
        recon = ET.parse(recon_files[0]).getroot()
        geo.sVoxel = np.array([_number(recon, name, True) for name in ("VOISizeZ", "VOISizeY", "VOISizeX")])
        matrix = _number(recon, "MatrixSize", True)
        if matrix != int(matrix):
            raise ValueError("Reconstruction MatrixSize must be an integer.")
        slices = int(np.floor(geo.sVoxel[0]/_number(recon, "SliceThickness", True)+.5))
        if slices < 1:
            raise ValueError("Reconstruction has no slices.")
        geo.nVoxel = np.array([slices, int(matrix), int(matrix)])
        geo.dVoxel = geo.sVoxel/geo.nVoxel
    else:
        geo.dVoxel = np.array([geo.dDetector[0], geo.dDetector[1], geo.dDetector[1]])*geo.DSO/geo.DSD
        width = int(np.ceil(geo.nDetector[1]+abs(offset)/geo.dDetector[1]))
        geo.nVoxel = np.array([geo.nDetector[0], width, width])
        geo.sVoxel = geo.nVoxel*geo.dVoxel
    return geo, scan


def _property(frame, name, positive=False):
    try:
        value = float(frame.properties[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"XIM requires numeric {name} property.") from exc
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid XIM {name} property.")
    return value


def _frames(paths, shape, threshold=0, require_angle=True):
    images = np.empty((len(paths), *shape), dtype=np.float32)
    angles, readings = [], []
    for path in paths:
        frame = read_xim(path)
        if not frame.pixels.size:
            continue
        if frame.pixels.shape != tuple(shape):
            raise ValueError(f"XIM dimensions disagree with Scan.xml: {path}")
        angle = (_property(frame, "GantryRtn") if require_angle or "GantryRtn" in frame.properties else 0)+90
        # Use shortest angular distance at the +/-180-degree boundary.
        if angles and threshold and abs((angle-angles[-1]+180) % 360-180) <= threshold:
            continue
        images[len(angles)] = np.fliplr(frame.pixels)
        angles.append(angle)
        readings.append(_property(frame, "KVNormChamber", True))
    if not angles:
        raise ValueError("No nonempty XIM frames were found.")
    return images[:len(angles)], np.array(angles), np.array(readings)


def _blank_files(folder, scan):
    acquisition = _element(scan, "Acquisitions")
    version = scan.attrib.get("Version", "")
    if "2.0" not in version and "2.7" not in version:
        raise NotImplementedError(f"Automatic blank selection supports TrueBeam 2.0/2.7, not {version!r}.")
    bowtie = "none" not in _text(acquisition, "Bowtie").lower()
    cc = _number(acquisition, "StopAngle") > _number(acquisition, "StartAngle")
    pattern = ("FilterBowtie.xim" if "2.0" in version else
               f"FilterBowtie_{'CC' if cc else 'CW'}*.xim") if bowtie else "Filter.xim"
    files = sorted(path for air in (folder / "Calibrations").glob("AIR-*")
                   for path in air.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} blank calibration under Calibrations/AIR-*.")
    if "2.0" in version and len(files) != 1:
        raise ValueError("TrueBeam 2.0 requires one unambiguous blank calibration.")
    return files


def log_normalize(projections, source_angles_deg, airnorm, blanks, blank_angles_deg, blank_airnorm):
    """CPU form of MATLAB LogNormal, including its adjacent-chamber mean.

    Angles for blanks are GantryRtn; projection source angles are GantryRtn+90.
    Blanks are linearly interpolated with endpoint clamping. Nonfinite and
    negative line integrals become zero, matching EnforcePositive.
    """
    projections = np.asarray(projections)
    blanks = np.asarray(blanks)
    source_angles_deg = np.asarray(source_angles_deg)
    airnorm, blank_airnorm = np.asarray(airnorm), np.asarray(blank_airnorm)
    if projections.ndim != 3 or blanks.ndim != 3 or projections.shape[1:] != blanks.shape[1:]:
        raise ValueError("Projection/blank detector shapes must match.")
    if not len(projections) or not len(blanks) or source_angles_deg.ndim != 1 or not np.isfinite(source_angles_deg).all():
        raise ValueError("Nonempty frames and finite source angles are required.")
    if airnorm.ndim != 1 or blank_airnorm.ndim != 1:
        raise ValueError("Chamber readings must be vectors.")
    if len(source_angles_deg) != len(projections) or len(airnorm) != len(projections) or len(blank_airnorm) != len(blanks):
        raise ValueError("Frame counts must match angles/chamber readings.")
    if not np.isfinite(airnorm).all() or not np.isfinite(blank_airnorm).all() or np.any(airnorm <= 0) or np.any(blank_airnorm <= 0):
        raise ValueError("Chamber readings must be finite and positive.")
    order = None
    if len(blanks) > 1:
        blank_angles_deg = np.asarray(blank_angles_deg)
        if len(blank_angles_deg) != len(blanks) or not np.isfinite(blank_angles_deg).all():
            raise ValueError("Blank angles must match blank frames.")
        order = np.argsort(blank_angles_deg)
        grid = blank_angles_deg[order]
        if np.any(np.diff(grid) <= 0):
            raise ValueError("Blank calibration angles must be distinct.")
    result = np.empty(projections.shape, dtype=np.float32)
    for k, angle in enumerate(source_angles_deg-90):
        if order is None:
            blank, reading = blanks[0], blank_airnorm[0]
        else:
            upper = int(np.clip(np.searchsorted(grid, angle), 1, len(grid)-1))
            lower = upper-1
            weight = np.clip((angle-grid[lower])/(grid[upper]-grid[lower]), 0, 1)
            left, right = order[lower], order[upper]
            blank = (1-weight)*blanks[left]+weight*blanks[right]
            reading = .5*(blank_airnorm[left]+blank_airnorm[right])
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.log((airnorm[k]/reading)*blank/(projections[k]+np.finfo(float).eps)+np.finfo(float).eps)
        result[k] = np.maximum(np.nan_to_num(value, nan=0, posinf=0, neginf=0), 0)
    return result


def VarianDataLoader(datafolder, *, acdc=True, dps=False, sc=False, bh=False,
                     load_geometry=True, ring_removal=True, ring_kernel=9):
    """Return float32 projections (views, v, u), Python geo and angles (rad).

    This CPU baseline intentionally defaults dps/sc/bh to False. MATLAB's
    default dps/sc corrections are not silently approximated. Pass True to
    request an explicit unsupported error. No GPU initialization/reset occurs.
    """
    for name, enabled in (("dps", dps), ("sc", sc), ("bh", bh)):
        if enabled:
            raise NotImplementedError(f"Varian {name} correction is not ported; use MATLAB's corrected MAT export.")
    folder = Path(datafolder).expanduser()
    geo, scan = geometry_from_xml(folder, load_geometry=load_geometry)
    acquisition = _element(scan, "Acquisitions")
    threshold = .9*abs(_number(acquisition, "Velocity"))/_number(acquisition, "FrameRate", True) if acdc else 0
    files = sorted(folder.rglob("Proj_*.xim"))
    if not files:
        raise FileNotFoundError(f"No Proj_*.xim projection files in {folder}")
    proj, angles, chamber = _frames(files, geo.nDetector, threshold)
    blank_paths = _blank_files(folder, scan)
    blanks, blank_angles, blank_chamber = _frames(blank_paths, geo.nDetector, require_angle=len(blank_paths)>1)
    output = log_normalize(proj, angles, chamber, blanks, blank_angles-90, blank_chamber)
    if ring_removal:
        if not isinstance(ring_kernel, (int, np.integer)) or ring_kernel < 1 or ring_kernel % 2 != 1:
            raise ValueError("Ring median kernel must be a positive odd integer.")
        output = median_filter(output, size=(1, 1, ring_kernel), mode="constant", cval=0)
    warnings.warn("Varian CPU baseline omits DPS, object scatter and beam-hardening corrections; "
                  "it is not equivalent to MATLAB's default preprocessing.", UserWarning, stacklevel=2)
    return output.astype(np.float32, copy=False), geo, np.deg2rad(angles)
