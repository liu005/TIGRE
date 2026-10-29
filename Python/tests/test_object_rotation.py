"""Object poses preserve hardware, transmission checks and reconstruction frames."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tigre.utilities.detector_geometry_gui import build_modular_surface, DetectorGeometryBuilder
from tigre.utilities.geometry_io import save_geometry_yaml, load_geometry_file, surface_archive_arrays, load_geometry_archive
from tigre.utilities.geometry_recon_export import reconstruction_script_text
from tigre.utilities.object_motion import fixed_object_geometry, rotation_zyz, object_angles
from tigre.utilities.ct_physics import validate_transmission_geometry


def settings():
    return dict(layout="Piecewise linear", n_views=3, source_radius_mm=1000,
        source_z_mm=0, module_shape=(3, 4), pixel_spacing_mm=(3, 3),
        detector_radius_mm=536, column_gap=0, n_modules=1, angular_span_deg=0,
        transverse_centers_mm=[0], normal_offsets_mm=[0], panel_angles_deg=[0],
        z_offsets_mm=[0], fixed_hardware=True)


def volume():
    return SimpleNamespace(sVoxel=np.array([20., 40., 60.]), nVoxel=np.array([2, 4, 6]),
        offOrigin=np.array([1., 2., 3.]), object_angles_deg=np.array([[0, 0, 0], [30, 90, 60], [-20, 40, 90]]))


def test_fixed_hardware_and_inverse_rigid_transform():
    surface, geo = build_modular_surface(**settings()), volume()
    np.testing.assert_allclose(surface.source_positions_mm, np.tile(surface.source_positions_mm[0], (3, 1)))
    np.testing.assert_allclose(surface.pixel_centers_mm, np.repeat(surface.pixel_centers_mm[:1], 3, axis=0))
    transformed, fixed = fixed_object_geometry(surface, geo)
    for i, angles in enumerate(geo.object_angles_deg):
        rotation = rotation_zyz(angles)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(transformed.source_positions_mm[i] @ rotation.T + geo.offOrigin[::-1], surface.source_positions_mm[i], atol=1e-4)
        np.testing.assert_allclose(transformed.pixel_centers_mm[i] @ rotation.T + geo.offOrigin[::-1], surface.pixel_centers_mm[i], atol=1e-4)
    np.testing.assert_allclose(fixed.offOrigin, 0)
    assert validate_transmission_geometry(surface, geo)["crossing_samples"] > 0


@pytest.mark.parametrize("angles", [[], [[0, 1]], [[0, float('nan'), 0]], [[0, 0, 0]] * 2])
def test_invalid_angle_lists(angles):
    with pytest.raises(ValueError):
        object_angles(SimpleNamespace(object_angles_deg=angles), 3)


def test_rotation_yaml_archive_and_export(tmp_path):
    geo, params = volume(), settings()
    path = tmp_path / "rotation.yaml"
    save_geometry_yaml(path, settings=params, geometry=geo)
    loaded = load_geometry_file(path)
    np.testing.assert_allclose(loaded.geometry.object_angles_deg, geo.object_angles_deg)
    archive = tmp_path / "rotation.npz"
    np.savez(archive, **surface_archive_arrays(loaded.surface, geo))
    restored, _ = load_geometry_archive(archive)
    np.testing.assert_allclose(restored.object_angles_deg, geo.object_angles_deg)
    script = tmp_path / "recon.py"
    namespace = dict(__file__=str(script), __name__="exported")
    exec(compile(reconstruction_script_text(path, script), str(script), "exec"), namespace)
    fixed, actual = namespace["prepare_geometry"](path)
    expected, _ = fixed_object_geometry(loaded.surface, geo)
    np.testing.assert_allclose(actual.source_positions_mm, expected.source_positions_mm)
    np.testing.assert_allclose(actual.pixel_centers_mm, expected.pixel_centers_mm)
    np.testing.assert_allclose(fixed.offOrigin, 0)


@pytest.mark.parametrize("angles", [[[30, 40, 50]], [[0, 0, 0], [30, 90, 60], [-20, 40, 90]], [0, 30, 90]])
def test_rotation_matlab_bridge_round_trip(tmp_path, angles):
    from tigre.utilities.geometry_matlab_bridge import to_mat, from_mat
    geo = volume()
    geo.object_angles_deg = np.asarray(angles)
    path, mat, output = (tmp_path / name for name in ("object.yaml", "object.mat", "output.yaml"))
    save_geometry_yaml(path, settings=settings(), geometry=geo)
    to_mat(path, mat)
    from_mat(mat, output)
    restored = load_geometry_file(output)
    np.testing.assert_allclose(object_angles(restored.geometry, 3), object_angles(geo, 3))
    assert restored.settings["fixed_hardware"]


def test_renderer_rotates_box_while_hardware_stays_fixed():
    from tigre.utilities.visualization.surface_renderer import SurfaceRenderer
    fig = plt.figure()
    try:
        renderer = SurfaceRenderer(fig.add_subplot(111, projection="3d"), build_modular_surface(**settings()), geo=volume())
        vertices = renderer.volume._vec.copy()
        source = np.asarray(renderer.source._offsets3d).copy()
        renderer.update(1)
        assert not np.allclose(vertices, renderer.volume._vec)
        np.testing.assert_allclose(renderer.source._offsets3d, source)
    finally:
        plt.close(fig)


def test_gui_d18_and_explicit_angle_modes(tmp_path):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    try:
        gui = DetectorGeometryBuilder(root)
        assert gui._rotate_object_toggle.master == gui._entries["nVoxel"].master
        assert gui._rotate_object_toggle.master != gui._entries["movement_views"].master
        root.update()
        for toggle, phrase in ((gui._rotate_object_toggle, "initial pose"),
                               (gui._timing_toggle, "one frame per source pulse"),
                               (gui._volume_motion_toggle, "interpolate the volume center")):
            for event in ("<Enter>", "<FocusIn>"):
                toggle.event_generate(event)
                root.update()
                assert phrase in gui._help_text.get("1.0", "end")
        # Every Movement widget must occupy its own grid cells. Late-created
        # shared controls must not cover timing labels or motion readouts.
        for page in (gui._rotate_object_toggle.master, gui._entries["movement_views"].master):
            occupied = set()
            for widget in page.grid_slaves():
                info = widget.grid_info()
                cells = {(r, c)
                         for r in range(int(info["row"]), int(info["row"]) + int(info["rowspan"]))
                         for c in range(int(info["column"]), int(info["column"]) + int(info["columnspan"]))}
                assert not occupied.intersection(cells), f"Overlapping control: {widget}"
                occupied.update(cells)
        gui._example.set("TIGRE d18: arbitrary object rotation (ZYZ)")
        gui.load_example()
        assert gui._rotate_object.get()
        assert gui._object_angle_mode.get() == "ZYZ angle list"
        with patch("tkinter.messagebox.showerror") as error:
            gui.confirm_parameters()
            error.assert_not_called()
        angles = gui._builder_geometry.object_angles_deg
        assert angles.shape == (100, 3)
        phase = np.linspace(0, 2 * np.pi, 100)
        # Inverse object poses must reproduce the demo's gantry poses.
        for pose, t in zip(angles, phase):
            expected = rotation_zyz(np.rad2deg([np.pi * np.sin(t), t, t]))
            np.testing.assert_allclose(rotation_zyz(pose).T, expected, atol=1e-14)
        # Compare against TIGRE's own projector geometry convention, not only
        # the Euler helper used by the GUI.
        import tigre
        from tigre.utilities.detector_geometry import surface_from_geometry
        reference_geo = tigre.geometry_default(high_resolution=False)
        reference_geo.nDetector = np.array([128, 128])
        reference_geo.dDetector = np.array([3.2, 3.2])
        reference_geo.sDetector = reference_geo.nDetector * reference_geo.dDetector
        reference = surface_from_geometry(reference_geo, np.column_stack((np.pi * np.sin(phase), phase, phase)))
        actual, _ = fixed_object_geometry(gui.surface, gui._builder_geometry)
        np.testing.assert_allclose(actual.source_positions_mm, reference.source_positions_mm, atol=0.001)
        np.testing.assert_allclose(actual.pixel_centers_mm, reference.pixel_centers_mm, atol=0.001)
        gui._rotate_object.set(False)
        gui._update_field_states()
        gui.confirm_parameters()
        assert not hasattr(gui._builder_geometry, "object_angles_deg")
        np.testing.assert_allclose(gui.surface.source_positions_mm, actual.source_positions_mm, atol=0.001)
        np.testing.assert_allclose(gui.surface.pixel_centers_mm, actual.pixel_centers_mm, atol=0.001)
        path = tmp_path / "d18-rig.yaml"
        save_geometry_yaml(path, settings=gui._read_parameters(), geometry=gui._read_volume())
        gui.load_geometry_path(path)
        assert not gui._rotate_object.get()
        np.testing.assert_allclose(gui.surface.source_positions_mm, actual.source_positions_mm, atol=0.001)
        gui._rotate_object.set(True)
        gui._update_field_states()
        gui._object_angle_mode.set("Start/end z angles")
        gui._object_angle_fields["object_start_angle"].set("90")
        gui._object_angle_fields["object_end_angle"].set("-90")
        np.testing.assert_allclose(gui._read_volume().object_angles_deg[:, 0], np.linspace(90, -90, 100))
        gui._example.set("Circular CT: curved modular detector")
        gui.load_example()
        assert not gui._rotate_object.get()
    finally:
        root.destroy()
