"""Projection-domain dual energy, using the calibration tables shared with MATLAB.

Inputs are finite, already log-normalized line integrals, ordered
(views, rows, columns); angles and matching tolerances are radians. Basis
thicknesses are in mm and the shared attenuation coefficients are in mm^-1.
Calibration coefficients are scanner/energy/material specific, not universal.
"""
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.interpolate import CubicSpline


def _table(name, override):
    if override is None:
        candidates = [Path(__file__).resolve().parents[3] / "Common" / "data" / name,
                      Path(sys.prefix) / "data" / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot find {name}; supply an explicit table path.")
    else:
        path = Path(override)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _real_array(value, name, convert=True):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real numeric array.")
    return array.astype(np.float64, copy=False) if convert else array


def _scan(projections, angles, name):
    projections = _real_array(projections, name, convert=False)
    angles = _real_array(angles, f"{name} angles")
    if projections.ndim != 3 or min(projections.shape) < 1:
        raise ValueError(f"{name} must have shape (views, rows, columns).")
    if angles.ndim != 1 or len(angles) != len(projections):
        raise ValueError(f"{name} angles must be a vector matching its views.")
    return projections, angles


def _interpolate(projections, angles, target, model, tolerance):
    nearest = int(np.argmin(abs(angles-target)))
    if model == "skip" and abs(angles[nearest]-target) > tolerance:
        return None
    if model in ("nearest", "skip"):
        return projections[nearest].astype(np.float64, copy=False)
    order = np.argsort(angles, kind="stable")
    x = angles[order]
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Linear interpolation requires two distinct, unique angles.")
    upper = int(np.clip(np.searchsorted(x, target), 1, len(x)-1))
    lower = upper-1
    weight = np.clip((target-x[lower])/(x[upper]-x[lower]), 0, 1)
    return ((1-weight)*projections[order[lower]].astype(np.float64, copy=False)
            + weight*projections[order[upper]].astype(np.float64, copy=False))


def interpolate_projection(projections, angles, target, model="nearest", tolerance=1.0):
    """Match one view on a continuous angle axis; clamp outside endpoints.

    ``skip`` returns None if the closest view exceeds tolerance (radians).
    Descending angles are supported; ties choose the first input view.
    No implicit wrapping or helix/motion compensation is performed.
    """
    projections, angles = _scan(projections, angles, "projections")
    if model not in ("nearest", "linear", "skip"):
        raise ValueError("Interpolation must be nearest, linear or skip.")
    if not np.isscalar(target) or not np.isfinite(target):
        raise ValueError("Target angle must be a finite scalar.")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be a finite nonnegative scalar in radians.")
    return _interpolate(projections, angles, float(target), model, float(tolerance))


def _polynomial(low, high, coefficients):
    c = _real_array(coefficients, "calibration coefficients")
    if c.shape != (9,):
        raise ValueError("Each material requires nine cubic transfer coefficients.")
    # L,H,L²,LH,H²,L³,HL²,LH²,H³ -- same order as MATLAB DeDecompose.
    return (low*(c[0]+low*(c[2]+c[5]*low))
            + high*(c[1]+high*(c[4]+c[8]*high))
            + low*high*(c[3]+c[6]*low+c[7]*high))


def decompose_dual_energy(proj_l, angles_l, proj_h, angles_h, *,
                          scanner="VarianTrueBeam", kv_low=80, kv_high=140,
                          material1="Al", material2="PMMA", interpolation="nearest",
                          tolerance=1.0, calibration_path=None):
    """Return two basis thickness stacks and retained high-energy angles.

    Low-energy views are matched/interpolated onto high-energy angles.
    In skip mode, unmatched views are removed from both outputs and angles.
    An all-skipped scan returns empty stacks. Negative fitted thicknesses
    are preserved, as in MATLAB; no positivity clipping is introduced.
    """
    low, low_angles = _scan(proj_l, angles_l, "low-energy projections")
    high, high_angles = _scan(proj_h, angles_h, "high-energy projections")
    if low.shape[1:] != high.shape[1:]:
        raise ValueError("Low/high detector shapes must match.")
    if interpolation not in ("nearest", "linear", "skip"):
        raise ValueError("Interpolation must be nearest, linear or skip.")
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be a finite nonnegative scalar in radians.")
    for value in (kv_low, kv_high):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0 or int(value) != value:
            raise ValueError("Calibration energies must be positive integer kVp values.")
    table = _table("dual_energy_calibrations.json", calibration_path)
    pair = f"V_{int(kv_low)}_{int(kv_high)}"
    try:
        calibration = table[scanner][pair]
        c, d = calibration[material1], calibration[material2]
    except KeyError as exc:
        raise ValueError(f"No calibration for {scanner}/{pair}/{material1}+{material2}.") from exc
    # Validate coefficients even if every projection is later skipped.
    _polynomial(0.0, 0.0, c); _polynomial(0.0, 0.0, d)
    first = np.empty(high.shape, dtype=np.float32)
    second = np.empty(high.shape, dtype=np.float32)
    keep = []
    for k, angle in enumerate(high_angles):
        matched = _interpolate(low, low_angles, angle, interpolation, tolerance)
        if matched is None:
            continue
        high_view = high[k].astype(np.float64, copy=False)
        first[len(keep)] = _polynomial(matched, high_view, c)
        second[len(keep)] = _polynomial(matched, high_view, d)
        keep.append(k)
    return first[:len(keep)], second[:len(keep)], high_angles[keep].copy()


def make_vm_projections(material1_proj, material2_proj, energy, *,
                        material1="Al", material2="PMMA", attenuation_path=None):
    """Synthesize dimensionless VM line integrals at energy in keV.

    Uses the same not-a-knot cubic spline as MATLAB MakeVMproj. Values
    outside the table range are extrapolated with a warning, as in MATLAB.
    """
    first = _real_array(material1_proj, "material1 projections", convert=False)
    second = _real_array(material2_proj, "material2 projections", convert=False)
    if first.shape != second.shape:
        raise ValueError("Basis projection shapes must match.")
    if not np.isscalar(energy) or not np.isfinite(energy) or energy <= 0:
        raise ValueError("Energy must be a finite positive scalar in keV.")
    table = _table("NIST_material_attenuation.json", attenuation_path)
    coefficients = []
    for material in (material1, material2):
        if material not in table:
            raise ValueError(f"Material {material} is not in the attenuation table.")
        x = _real_array(table[material]["energy"], "table energies")
        y = _real_array(table[material]["mu"], "table attenuation")
        if x.ndim != 1 or x.size < 2 or y.shape != x.shape or np.any(np.diff(x) <= 0):
            raise ValueError("Attenuation table energies must be strictly increasing.")
        if energy < x[0] or energy > x[-1]:
            warnings.warn(f"Energy outside the {material} table range; extrapolating.", RuntimeWarning, stacklevel=2)
        coefficients.append(float(CubicSpline(x, y)(energy)))
    result = np.empty(first.shape, dtype=np.float32)
    # Bound double-precision temporary memory for large scanner acquisitions.
    left, right, output = first.flat, second.flat, result.flat
    for start in range(0, first.size, 1_048_576):
        end = min(start+1_048_576, first.size)
        output[start:end] = (coefficients[0]*np.asarray(left[start:end], dtype=np.float64)
                             + coefficients[1]*np.asarray(right[start:end], dtype=np.float64))
    return result
