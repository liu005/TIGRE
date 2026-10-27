"""Synthetic XIM/XML integration tests; no scanner files or CUDA required."""
from pathlib import Path
import struct
import numpy as np
import pytest
from tigre.utilities.io import VarianDataLoader
from tigre.utilities.io.VarianDataLoader import geometry_from_xml, log_normalize
from tigre.utilities.io.xim import read_xim


def write_xim(path, image, *, compressed=False, properties=None, bpp=4):
    image = np.asarray(image, dtype=np.int32)
    height, width = image.shape
    out = bytearray(b"VMS.XI\0\0"+struct.pack("<6i", 1, width, height, bpp*8, bpp, int(compressed)))
    if compressed:
        flat = image.ravel().astype(np.int64)
        seed = min(width+1, flat.size)
        payload = bytearray(flat[:seed].astype("<i4").tobytes())
        codes = []
        for i in range(seed, len(flat)):
            delta = int(flat[i]-flat[i-1]-flat[i-width]+flat[i-width-1])
            code = 0 if -128 <= delta < 128 else 1 if -32768 <= delta < 32768 else 2
            codes.append(code)
            payload.extend(struct.pack(("<b", "<h", "<i")[code], delta))
        lookup = bytearray((len(codes)+3)//4)
        for i, code in enumerate(codes):
            lookup[i//4] |= code << (2*(i % 4))
        out.extend(struct.pack("<i", len(lookup))+lookup+struct.pack("<i", len(payload))+payload)
        out.extend(struct.pack("<i", image.size*bpp))
    else:
        dtype = "<i4" if bpp == 4 else f"<u{bpp}"
        payload = image.astype(dtype).tobytes()
        out.extend(struct.pack("<i", len(payload))+payload)
    out.extend(struct.pack("<2i", 0, len(properties or {})))
    for name, value in (properties or {}).items():
        encoded = name.encode()
        out.extend(struct.pack("<i", len(encoded))+encoded)
        if isinstance(value, str):
            encoded = value.encode()
            out.extend(struct.pack("<2i", 2, len(encoded))+encoded)
        elif isinstance(value, int):
            out.extend(struct.pack("<2i", 0, value))
        elif isinstance(value, np.ndarray):
            kind = 4 if value.dtype.kind == "f" else 5
            encoded = value.astype("<f8" if kind == 4 else "<i4").tobytes()
            out.extend(struct.pack("<2i", kind, len(encoded))+encoded)
        else:
            out.extend(struct.pack("<id", 1, value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def make_dataset(folder, version="2.0"):
    folder.mkdir(exist_ok=True)
    folder.joinpath("Scan.xml").write_text(f'''<Scan Version="{version}"><Acquisitions>
    <SID>150</SID><SAD>100</SAD><ImagerSizeX>3</ImagerSizeX><ImagerSizeY>2</ImagerSizeY>
    <ImagerResX>2</ImagerResX><ImagerResY>4</ImagerResY><ImagerLat>6</ImagerLat>
    <Velocity>1</Velocity><FrameRate>1</FrameRate><Bowtie>None</Bowtie>
    <StartAngle>0</StartAngle><StopAngle>180</StopAngle></Acquisitions></Scan>''')
    image = np.array([[10, 20, 25], [40, 50, 100]], dtype=np.int32)
    for k, angle in enumerate((0.0, .2, 1.5)):
        write_xim(folder / "Acquisitions" / f"Proj_{k:03d}.xim", image,
                  compressed=k % 2 == 0,
                  properties=dict(GantryRtn=angle, KVNormChamber=50.0))
    write_xim(folder / "Calibrations/AIR-test/Filter.xim", np.full((2, 3), 400),
              properties=dict(KVNormChamber=100.0))
    return image


@pytest.mark.parametrize("compressed", [False, True])
def test_xim_pixels_all_property_types_and_predictor_widths(tmp_path, compressed):
    image = np.array([[100, 101, 103, 106], [105, 108, 1000, 40000],
                      [200, 300, 1000000, 301], [22, 21, 20, 19]], dtype=np.int32)
    properties = dict(GantryRtn=20.5, KVNormChamber=10.0, integer=7,
                      text="test", doubles=np.array([1.2, 3.4]), integers=np.array([2, -3]))
    path = tmp_path / "frame.xim"
    write_xim(path, image, compressed=compressed, properties=properties)
    decoded = read_xim(path)
    np.testing.assert_array_equal(decoded.pixels, image)
    for name in properties:
        np.testing.assert_equal(decoded.properties[name], properties[name])


@pytest.mark.parametrize("bpp,values", [(1, [0, 128, 255]), (2, [0, 32768, 65535])])
def test_uncompressed_unsigned_pixels(tmp_path, bpp, values):
    path = tmp_path / "unsigned.xim"
    write_xim(path, [values], bpp=bpp)
    np.testing.assert_array_equal(read_xim(path).pixels, [values])


def test_corrupt_and_truncated_xim_raise(tmp_path):
    path = tmp_path / "bad.xim"
    write_xim(path, [[1, 2], [3, 4]], compressed=True)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="Truncated|property"):
        read_xim(path)
    path.write_bytes(b"not xim!")
    with pytest.raises(ValueError, match="supported XIM"):
        read_xim(path)


def test_loader_geometry_orientation_acdc_chamber_log_and_dtype(tmp_path):
    image = make_dataset(tmp_path)
    with pytest.warns(UserWarning, match="baseline"):
        projections, geo, angles = VarianDataLoader(tmp_path, ring_removal=False)
    assert projections.shape == (2, 2, 3) and projections.dtype == np.float32
    np.testing.assert_allclose(projections[0], np.log(200/np.fliplr(image)), rtol=1e-6)
    np.testing.assert_allclose(angles, np.deg2rad([90, 91.5]))
    np.testing.assert_array_equal(geo.nDetector, [2, 3])
    np.testing.assert_array_equal(geo.offDetector, [0, -6])
    np.testing.assert_array_equal(geo.nVoxel, [2, 6, 6])
    np.testing.assert_allclose(geo.dVoxel, [8/3, 4/3, 4/3])
    with pytest.warns(UserWarning):
        projections, _, angles = VarianDataLoader(tmp_path, acdc=False, ring_removal=False)
    assert len(projections) == len(angles) == 3


def test_reconstruction_xml_and_ambiguous_geometry(tmp_path):
    make_dataset(tmp_path)
    xml = '''<Reconstruction><VOISizeX>10</VOISizeX><VOISizeY>20</VOISizeY>
    <VOISizeZ>9</VOISizeZ><SliceThickness>2</SliceThickness><MatrixSize>8</MatrixSize></Reconstruction>'''
    tmp_path.joinpath("Reconstruction.xml").write_text(xml)
    geo, _ = geometry_from_xml(tmp_path)
    np.testing.assert_array_equal(geo.nVoxel, [5, 8, 8])
    np.testing.assert_array_equal(geo.sVoxel, [9, 20, 10])
    tmp_path.joinpath("other").mkdir()
    tmp_path.joinpath("other/Reconstruction.xml").write_text(xml)
    with pytest.raises(ValueError, match="Multiple Reconstruction"):
        geometry_from_xml(tmp_path)
    geo, _ = geometry_from_xml(tmp_path, load_geometry=False)
    np.testing.assert_array_equal(geo.nVoxel, [2, 6, 6])


def test_angle_dependent_blanks_match_matlab_normalization_formula():
    proj = np.full((2, 2, 3), 100, np.float32)
    blanks = np.array([np.full((2, 3), 200), np.full((2, 3), 600)])
    value = log_normalize(proj, [90, 95], [2, 2], blanks, [0, 10], [1, 3])
    np.testing.assert_allclose(value[:, 0, 0], np.log([2, 4]))
    reverse = log_normalize(proj, [90, 95], [2, 2], blanks[::-1], [10, 0], [3, 1])
    np.testing.assert_array_equal(value, reverse)


def test_truebeam27_directional_bowtie_blank_selection_and_ring_filter(tmp_path):
    make_dataset(tmp_path, "2.7")
    path = tmp_path / "Scan.xml"
    path.write_text(path.read_text().replace("<Bowtie>None</Bowtie>", "<Bowtie>Full</Bowtie>"))
    for k, (angle, count) in enumerate(((0.0, 200), (10.0, 600))):
        write_xim(tmp_path / f"Calibrations/AIR-test/FilterBowtie_CC{k}.xim", np.full((2, 3), count),
                  properties=dict(GantryRtn=angle, KVNormChamber=100.0))
    with pytest.warns(UserWarning):
        result, _, _ = VarianDataLoader(tmp_path, ring_kernel=3)
    assert result.shape == (2, 2, 3) and np.isfinite(result).all()


def test_dual_energy_demo_accepts_two_raw_varian_folders(tmp_path):
    import runpy
    demo = runpy.run_path(str(Path(__file__).resolve().parents[1] / "demos/d23_DualEnergy.py"))
    low_folder, high_folder = tmp_path / "low", tmp_path / "high"
    make_dataset(low_folder); make_dataset(high_folder)
    with pytest.warns(UserWarning):
        data = demo["load_varian_pair"](low_folder, high_folder, ring_removal=False)
    result = demo["process"](data)
    assert result["VM_proj"].shape == (2, 2, 3) and np.isfinite(result["VM_proj"]).all()


@pytest.mark.parametrize("flag", ["dps", "sc", "bh"])
def test_unsupported_corrections_are_explicit(tmp_path, flag):
    with pytest.raises(NotImplementedError, match=flag):
        VarianDataLoader(tmp_path, **{flag: True})


def test_missing_folder_scan_and_projections_have_actionable_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        VarianDataLoader(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="Scan.xml"):
        VarianDataLoader(tmp_path)
    make_dataset(tmp_path)
    for path in tmp_path.rglob("Proj_*.xim"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="Proj_"):
        VarianDataLoader(tmp_path)
