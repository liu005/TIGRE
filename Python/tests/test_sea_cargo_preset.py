"""Sea-cargo preset uses source-facing modules and ray-aligned projected seams."""
from unittest.mock import patch
import numpy as np
import pytest
from tigre.utilities.detector_geometry_gui import DetectorGeometryBuilder
from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file


def test_sea_cargo_preset_and_yaml(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        gui = DetectorGeometryBuilder(root)
        gui._example.set("Sea cargo: 20-foot container (2D Digital Radiography)")
        with patch("tkinter.messagebox.showerror") as error:
            gui.load_example()
            gui.confirm_parameters()
            error.assert_not_called()
        surface, geo = gui.surface, gui._builder_geometry
        assert surface.valid_mask.shape == (1253, 36, 112)
        np.testing.assert_allclose(surface.source_positions_mm,
            np.tile([4230*np.cos(np.deg2rad(13)), 0, -4230*np.sin(np.deg2rad(13))], (1253, 1)), atol=.001)
        pixels = surface.pixel_centers_mm
        np.testing.assert_array_equal(pixels[0], pixels[-1])
        source = surface.source_positions_mm[0].astype(float)
        edges = []
        assert len(surface._module_slices) == 28
        for rows, columns in surface._module_slices:
            grid = pixels[0, rows, columns].astype(float)
            assert grid.shape == (36, 4, 3)
            row = (grid[-1].mean(axis=0)-grid[0].mean(axis=0))/35
            column = (grid[:, -1].mean(axis=0)-grid[:, 0].mean(axis=0))/3
            np.testing.assert_allclose(np.linalg.norm(row), 5, atol=1e-4)
            np.testing.assert_allclose(np.linalg.norm(column), 5, atol=1e-4)
            normal = np.cross(column, row); normal /= np.linalg.norm(normal)
            toward_source = source-grid.mean(axis=(0, 1)); toward_source /= np.linalg.norm(toward_source)
            np.testing.assert_allclose(normal, toward_source, atol=2e-6)
            edges.append((grid[0].mean(axis=0)-row/2, grid[-1].mean(axis=0)+row/2))
        for previous, following in zip(edges, edges[1:]):
            a, b = previous[1]-source, following[0]-source
            np.testing.assert_allclose(a/np.linalg.norm(a), b/np.linalg.norm(b), atol=1e-7)
        # Joint after 18 vertical modules: near edge of the upper arm is
        # Raised horizontal support stays on the same joint-edge source ray.
        assert edges[18][0][0] < edges[17][1][0]
        assert edges[18][0][2]-edges[17][1][2] > 300
        from tigre.utilities.cargo_detector import cargo_module_layout
        centers, angles, vertical, horizontal = cargo_module_layout()
        baseline, _, baseline_vertical, _ = cargo_module_layout(horizontal_lift_mm=0)
        np.testing.assert_allclose(centers[:vertical], baseline[:baseline_vertical])
        np.testing.assert_allclose(centers[vertical:, 2], baseline[baseline_vertical, 2]+300, atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(source), 4230, atol=.001)
        assert (vertical, horizontal) == (18, 10)
        np.testing.assert_allclose(centers[:vertical, 0], centers[0, 0])
        np.testing.assert_allclose(centers[vertical:, 2], centers[vertical, 2])
        # Check every fan direction across the complete x-z silhouette,
        # including seams. Conveyor position does not alter this silhouette.
        corners = np.array([[x, 0, z] for x in (-1220, 1220) for z in (-1195, 1395)])
        rays = corners-source
        corner_angles = np.arctan2(rays[:, 2], -rays[:, 0])
        phi = np.linspace(corner_angles.min(), corner_angles.max(), 4097)
        directions = np.column_stack((-np.cos(phi), np.zeros(len(phi)), np.sin(phi)))
        covered = np.zeros(len(phi), dtype=bool)
        for center, angle in zip(centers, angles):
            beta = np.deg2rad(angle)
            normal = np.array([np.cos(beta), 0, -np.sin(beta)])
            row = np.array([np.sin(beta), 0, np.cos(beta)])
            distance = ((center-source) @ normal) / (directions @ normal)
            along = (source+distance[:, None]*directions-center) @ row
            covered |= (distance > 0) & (np.abs(along) <= 90.001)
        assert covered.all()
        from tigre.utilities.ct_physics import _segment_hits_box
        lo, hi = np.array([-1220, -3030, -1195]), np.array([1220, 3030, 1395])
        assert not _segment_hits_box(source, pixels[0, :18, :4].reshape(-1, 3), lo, hi).any()
        assert not _segment_hits_box(source, pixels[0, :, -4:].reshape(-1, 3), lo, hi).any()

        np.testing.assert_array_equal(geo.nVoxel, [518, 1212, 488])
        np.testing.assert_allclose(geo.dVoxel, 5)
        np.testing.assert_allclose(np.diff(geo.offOrigin[:, 1]), 5)
        np.testing.assert_allclose(np.abs(geo.offOrigin[[0, -1], 1])-geo.sVoxel[1]/2, 100)
        np.testing.assert_allclose(geo.offOrigin, np.column_stack((np.full(1253, 100), np.linspace(-3130, 3130, 1253), np.zeros(1253))))
        assert gui._physical_report["crossing_samples"] > 0
        assert not gui._rotate_object.get() and gui._volume_motion.get()
        # Fixed-hardware DR timing ignores orbit/z-speed controls and derives
        # a vector velocity from endpoints, including conveyor motion along y.
        gui._timing_enabled.set(True)
        gui._update_field_states()
        assert str(gui._entries["rotation_period_s"].cget("state")) == "disabled"
        assert str(gui._entries["table_speed_mm_s"].cget("state")) == "disabled"
        gui._timing_fields["rotation_period_s"].set("unused in DR")
        gui._timing_fields["table_speed_mm_s"].set("unused in DR")
        with patch("tkinter.messagebox.showerror") as error:
            gui.confirm_parameters()
            error.assert_not_called()
        geo = gui._builder_geometry
        np.testing.assert_allclose(geo.view_times_s[-1], 12.52)
        np.testing.assert_allclose((geo.offOrigin[-1]-geo.offOrigin[0])/geo.view_times_s[-1], [0, 500, 0])
        np.testing.assert_array_equal(gui.surface.source_positions_mm, surface.source_positions_mm)
        np.testing.assert_array_equal(gui.surface.pixel_centers_mm, pixels)
        assert "speed 500" in gui._timing_summary.get()
        assert "Turns" not in gui._timing_summary.get()
        path = tmp_path / "sea-cargo.yaml"
        save_geometry_yaml(path, settings=gui._read_parameters(), geometry=geo)
        document = load_geometry_file(path)
        assert document.geometry.simulation_mode == "dr"
        np.testing.assert_allclose(document.geometry.view_times_s, geo.view_times_s)
        from tigre.utilities.geometry_io import surface_archive_arrays, load_geometry_archive
        archive = tmp_path / "cargo.npz"
        np.savez_compressed(archive, **surface_archive_arrays(surface, geo))
        archived_geo, _ = load_geometry_archive(archive)
        assert archived_geo.simulation_mode == "dr"
        np.testing.assert_array_equal(document.surface.pixel_centers_mm, pixels)
        np.testing.assert_allclose(document.geometry.offOrigin, geo.offOrigin)
        gui.load_geometry_path(path)
        assert gui._volume_motion.get()
        np.testing.assert_allclose(gui._builder_geometry.offOrigin, geo.offOrigin)
        from tigre.utilities.geometry_recon_export import reconstruction_script_text
        script = reconstruction_script_text(path, tmp_path / "cargo_dr.py")
        assert "cgls" not in script.lower() and "fdk" not in script.lower()
        namespace = dict(__name__="exported_dr", __file__=str(tmp_path / "cargo_dr.py"))
        exec(compile(script, "cargo_dr.py", "exec"), namespace)
        output = tmp_path / "dr.npz"
        samples = np.broadcast_to(np.repeat(np.arange(1, 29), 4)[None, None, :], (1253, 36, 112)).astype(np.float32)
        with patch("tigre.Ax_surface", return_value=samples) as forward:
            namespace["main"](["--voxels", "8", "12", "6", "--output", str(output)])
        assert forward.call_args.args[0].shape == (8, 12, 6)
        with np.load(output) as result:
            np.testing.assert_allclose(result["transmission"], np.exp(-samples), rtol=1e-6)
            assert result["dr_image"].shape == (1008, 1253)
            np.testing.assert_allclose(result["dr_image"][:, 0], np.repeat(np.arange(1, 29), 36))
            assert result["module_slices"].shape == (28, 4)
            np.testing.assert_allclose(result["view_times_s"][-1], 12.52)
        gui._volume_fields["offOrigin"].set("10, -3130, 20")
        gui._volume_end.set("110, 3130, 220")
        moving = gui._read_volume()
        np.testing.assert_allclose(moving.offOrigin, np.linspace([10, -3130, 20], [110, 3130, 220], 1253))
        gui._timing_fields["firing_rate_hz"].set("200")
        with pytest.raises(ValueError, match="exceeds"):
            gui._read_parameters()
        gui._timing_fields["firing_rate_hz"].set("100")
        gui._fields["source_end_mm"].set("4000, 0, 0")
        with pytest.raises(ValueError, match="DR timing requires"):
            gui._read_parameters()
    finally:
        root.destroy()
