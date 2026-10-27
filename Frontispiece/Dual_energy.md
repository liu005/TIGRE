# Dual-energy projection processing

Python demo 23 ports MATLAB demo 23's projection-domain workflow using the same
`Common/data/dual_energy_calibrations.json` and
`Common/data/NIST_material_attenuation.json`. Bundled calibration pairs are
VarianTrueBeam 80/140 and 90/140 kVp with Al/PMMA basis thicknesses. Coefficients
are specific to scanner calibration; they do not calibrate arbitrary scanners.
The acquisition folders in MATLAB d23 are placeholders. No measured low/high
projection pair is included with the demo.

## Use the same data processed by MATLAB d23

After loading the two actual scans in MATLAB, with beam-hardening correction
disabled as in the demo, export their preprocessed projections and geometry:

```matlab
save('dual_energy_input.mat','proj_l','angles_l','proj_h','angles_h','geo','-v7');
```

From the repository's `Python/` folder:

```powershell
conda run -n recon python demos/d23_DualEnergy.py dual_energy_input.mat --energy 60
```

The reader converts MATLAB rows/columns/views to Python views/rows/columns,
xyz volume metadata to zyx, and u/v detector metadata to v/u. Local detector
rotation components remain x/y/z. Angles remain radians. Use MATLAB `-v7`;
SciPy's reader does not load MATLAB v7.3/HDF5 workspaces.

The default reconstructs VM projections with FDK and 100 OS-SART iterations
and displays both. `--no-reconstruct` runs the material decomposition and VM
synthesis on CPU; `--output results.npz` saves numeric arrays and angles.

```powershell
conda run -n recon python demos/d23_DualEnergy.py --synthetic --output smoke.npz
```

This last command exercises the transfer functions with synthetic line
integrals. It is not a physical dual-spectrum simulation or an acquisition.

## Python API and matching

```python
from tigre.utilities.dual_energy import decompose_dual_energy, make_vm_projections

al, pmma, angles = decompose_dual_energy(
    proj_l, angles_l, proj_h, angles_h,
    scanner="VarianTrueBeam", kv_low=80, kv_high=140,
    material1="Al", material2="PMMA", interpolation="nearest")
vm = make_vm_projections(al, pmma, 60)
```

Inputs are finite, log-normalized line integrals with shape
`(views, rows, columns)`. The cubic terms, in coefficient order, are
`L, H, L², LH, H², L³, HL², LH², H³`. Outputs are fitted equivalent material
thicknesses in mm; negative fitted thicknesses are preserved. VM output is
dimensionless line integral using the table's linear attenuation in mm⁻¹.
Processing uses per-view/chunked double temporaries and float32 outputs.

`nearest` matches low-energy views onto high-energy angles. `linear` sorts
unique low-energy angles and clamps outside their endpoints. `skip` excludes
views whose absolute nearest-angle distance exceeds `tolerance`, in radians;
both basis stacks and the returned angles are shortened together. No implicit
angular wrapping or motion correction is performed. Provide both scans on the
same continuous angle axis. Custom calibration and attenuation JSON paths are
supported by the API. VM synthesis uses not-a-knot cubic splines, matching
MATLAB `spline`; out-of-range energies extrapolate with a warning.

## Raw Varian folders

```powershell
conda run -n recon python demos/d23_DualEnergy.py --low-dir LOW --high-dir HIGH --no-reconstruct
```

This uses the [CPU Varian baseline](Varian_loader.md), which omits DPS/object
scatter and beam-hardening corrections. Its preprocessing is not equivalent
to MATLAB's default Varian pipeline. For comparison with MATLAB d23, use the
preprocessed MAT export above.

## Shared fixes and validation

MATLAB data lookup is now relative to its utility files rather than the current
working directory. Its kVp option parsing and linear/skip angle interpolation
are corrected. The shared attenuation energy grid had `10` between 80 and
110 keV; that label is corrected to 100, retaining attenuation values. The
100-keV interpretation agrees with the NIST
[Al table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html)
and [PMMA table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/pmma.html).

`test_dual_energy.py` covers the shared coefficients, matching, skipped/empty
outputs, attenuation nodes, validation and MATLAB import. MATLAB
`test_dualEnergy` covers both kVp pairs and VM/interpolation. Live comparisons
check Python results against MATLAB output, including VM at 67 keV between
table nodes. No real paired scanner dataset has been tested locally.
