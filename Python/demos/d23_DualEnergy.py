# %% Demo 23: Dual-energy CBCT
"""Port MATLAB d23 using its exact shared calibration and attenuation tables.

MATLAB d23's acquisition folders are placeholders, not bundled scans. After
loading its paired scans (with beam-hardening correction disabled), export:
    save('dual_energy_input.mat','proj_l','angles_l','proj_h','angles_h','geo','-v7')
Then in recon:
    python demos/d23_DualEnergy.py dual_energy_input.mat --energy 60
Use --no-reconstruct for CPU-only processing, or --synthetic for a CPU smoke
example of the transfer functions (not a calibrated synthetic scanner).
Alternatively use --low-dir LOW --high-dir HIGH for the CPU Varian baseline;
it omits MATLAB's default DPS/object-scatter corrections and warns explicitly.
"""
import argparse
from pathlib import Path
import sys

# Direct IDE launches otherwise prefer an older site-packages TIGRE copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tigre
from tigre.utilities.dual_energy import decompose_dual_energy, make_vm_projections


def load_matlab_input(path):
    """Read the MATLAB d23 workspace; convert xyz/u-v axes to Python order."""
    from scipy.io import loadmat
    data = loadmat(path, struct_as_record=False, squeeze_me=False)
    required = {"proj_l", "angles_l", "proj_h", "angles_h"}
    if not required.issubset(data):
        raise ValueError(f"MAT file requires {sorted(required)}.")
    result = {}
    for key in ("proj_l", "proj_h"):
        array = np.asarray(data[key])
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError(f"{key} must be MATLAB rows x columns x views.")
        result[key] = np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
    for key in ("angles_l", "angles_h"):
        result[key] = np.asarray(data[key], dtype=np.float64).ravel()
    result["geo"] = None
    if "geo" in data:
        raw = data["geo"][0, 0]
        geo = tigre.geometry()
        for key in ("nVoxel", "sVoxel", "dVoxel", "nDetector", "dDetector", "sDetector"):
            value = np.asarray(getattr(raw, key)).ravel()[::-1].copy()
            expected = 3 if "Voxel" in key else 2
            if value.size != expected or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"Invalid MATLAB geometry field {key}.")
            if key.startswith("n") and np.any(value != np.floor(value)):
                raise ValueError(f"MATLAB {key} counts must be integers.")
            setattr(geo, key, value.astype(int) if key.startswith("n") else value)
        for key in ("DSO", "DSD", "accuracy", "COR"):
            if key in ("DSO", "DSD") and not hasattr(raw, key):
                raise ValueError(f"MATLAB geometry requires {key}.")
            value = np.asarray(getattr(raw, key, 0 if key == "COR" else 0.5)).ravel()
            if not np.isfinite(value).all() or (key != "COR" and np.any(value <= 0)):
                raise ValueError(f"Invalid MATLAB geometry field {key}.")
            setattr(geo, key, value.item() if value.size == 1 else value)
        for key, components, reverse in (("offOrigin", 3, True), ("offDetector", 2, True),
                                         ("rotDetector", 3, False)):
            value = np.asarray(getattr(raw, key, np.zeros((components, 1))))
            value = value.reshape(components, -1).T
            if reverse:
                value = value[:, ::-1]
            setattr(geo, key, value[0].copy() if len(value) == 1 else value.copy())
        geo.mode = str(np.asarray(getattr(raw, "mode", "cone")).ravel()[0])
        if tuple(geo.nDetector) != result["proj_h"].shape[1:]:
            raise ValueError("MATLAB geo.nDetector must match projection rows/columns.")
        result["geo"] = geo
    return result


def process(data, *, energy=60.0, kv_low=80, kv_high=140,
            scanner="VarianTrueBeam", interpolation="nearest", tolerance=1.0):
    """Perform MATLAB's Al/PMMA decomposition and VM projection synthesis."""
    first, second, angles = decompose_dual_energy(
        data["proj_l"], data["angles_l"], data["proj_h"], data["angles_h"],
        scanner=scanner, kv_low=kv_low, kv_high=kv_high,
        interpolation=interpolation, tolerance=tolerance)
    vm = make_vm_projections(first, second, energy)
    return dict(Al_proj=first, PMMA_proj=second, VM_proj=vm, angles=angles)


def load_varian_pair(low_folder, high_folder, *, acdc=True, ring_removal=True):
    """Load paired raw acquisitions using the explicitly limited CPU baseline."""
    from tigre.utilities.io import VarianDataLoader
    low, low_geo, low_angles = VarianDataLoader(low_folder, acdc=acdc,
        ring_removal=ring_removal, dps=False, sc=False, bh=False)
    high, high_geo, high_angles = VarianDataLoader(high_folder, acdc=acdc,
        ring_removal=ring_removal, dps=False, sc=False, bh=False)
    for name in ("DSO", "DSD", "nDetector", "dDetector", "offDetector"):
        if not np.allclose(getattr(low_geo, name), getattr(high_geo, name)):
            raise ValueError(f"Low/high Varian geometry differs in {name}.")
    return dict(proj_l=low, angles_l=low_angles, proj_h=high, angles_h=high_angles, geo=high_geo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, help="MATLAB -v7 workspace from demo 23")
    parser.add_argument("--synthetic", action="store_true", help="CPU transfer-function smoke example")
    parser.add_argument("--low-dir", type=Path, help="Raw low-energy Varian acquisition (baseline corrections)")
    parser.add_argument("--high-dir", type=Path, help="Raw high-energy Varian acquisition (baseline corrections)")
    parser.add_argument("--energy", type=float, default=60.0, help="VM energy in keV")
    parser.add_argument("--kv-low", type=int, default=80)
    parser.add_argument("--kv-high", type=int, default=140)
    parser.add_argument("--scanner", default="VarianTrueBeam")
    parser.add_argument("--interpolation", choices=("nearest", "linear", "skip"), default="nearest")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Skip matching tolerance in radians")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--no-reconstruct", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional numeric output NPZ")
    args = parser.parse_args(argv)
    if bool(args.low_dir) != bool(args.high_dir):
        parser.error("Provide both --low-dir and --high-dir.")
    if sum((args.synthetic, args.input is not None, args.low_dir is not None)) != 1:
        parser.error("Provide a MATLAB input file, paired --low-dir/--high-dir, or --synthetic.")
    if args.iterations < 1:
        parser.error("Iterations must be positive.")
    if args.synthetic:
        low = np.linspace(0, 1, 6*8*10, dtype=np.float32).reshape(6, 8, 10)
        data = dict(proj_l=low, proj_h=low*.8, angles_l=np.arange(6)*.1,
                    angles_h=np.arange(6)*.1, geo=None)
    elif args.low_dir:
        data = load_varian_pair(args.low_dir, args.high_dir)
    else:
        data = load_matlab_input(args.input)
    result = process(data, energy=args.energy, kv_low=args.kv_low, kv_high=args.kv_high,
                     scanner=args.scanner, interpolation=args.interpolation, tolerance=args.tolerance)
    if not len(result["angles"]):
        raise ValueError("No high-energy views matched the chosen tolerance.")
    if not args.no_reconstruct and not args.synthetic:
        import tigre.algorithms as algs
        from copy import deepcopy
        geo = deepcopy(data["geo"])
        if geo is None:
            raise ValueError("Reconstruction requires exported geo; use --no-reconstruct otherwise.")
        # Skip mode removes matching geometry rows as well as measured views.
        keep = np.isin(data["angles_h"], result["angles"])
        for name in ("offOrigin", "offDetector", "rotDetector", "COR", "DSO", "DSD"):
            value = np.asarray(getattr(geo, name))
            if value.ndim > 1 and value.shape[0] == len(keep):
                setattr(geo, name, value[keep])
            elif name in ("COR", "DSO", "DSD") and value.ndim == 1 and len(value) == len(keep):
                setattr(geo, name, value[keep])
        result["VMimgFDK"] = algs.fdk(result["VM_proj"], geo, result["angles"])
        result["VMimgOSSART"] = algs.os_sart(result["VM_proj"], geo, result["angles"], args.iterations)
        tigre.plotimg(np.concatenate([result["VMimgFDK"], result["VMimgOSSART"]], axis=2), dim="z")
    if args.output:
        np.savez_compressed(args.output, **result)
    print(f"Processed {len(result['angles'])} views at {args.energy:g} keV with MATLAB's shared tables.")
    return result


if __name__ == "__main__":
    main()
