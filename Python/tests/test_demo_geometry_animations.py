"""Check animation placement and render representative demo 17/18 frames."""
import ast
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from test_detector_geometry_examples import DEMOS, _demo_geometry
from tigre.utilities.detector_geometry import surface_from_geometry
from tigre.utilities.visualization.animate_geometry import animate_geometry


@pytest.mark.parametrize('filename', ['d17_DetectorRotation.py', 'd18_ArbitraryAxisOfRotation.py'])
def test_animation_immediately_follows_projection_plot_and_renders(filename):
    nodes = ast.parse((DEMOS / filename).read_text()).body
    index = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'plotproj')
    assert isinstance(nodes[index+1], ast.Assign)
    assert nodes[index+1].targets[0].id == 'geometry_animation'
    demo = _demo_geometry(filename, 'head')
    surface = surface_from_geometry(demo['geo'], demo['angles'])
    if filename.startswith('d18'):
        assert np.ptp(surface.source_positions_mm[:, 2]) > 900
    animation = animate_geometry(demo['geo'], surface=surface)
    try:
        animation._fig.canvas.draw()
        for frame in (0, 50, 99):
            animation._func(frame)
            animation._fig.canvas.draw()
            assert f'view {frame}/100' in animation._fig.axes[0].get_title()
    finally:
        animation._draw_was_started = True
        plt.close(animation._fig)


def test_d19_animation_follows_projection_plot_and_detector_stays_fixed():
    nodes = ast.parse((DEMOS / 'd19_DBTexample.py').read_text()).body
    index = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'plotproj')
    assert nodes[index+1].targets[0].id == 'geometry_animation'
    stop = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'projections' for t in node.targets))
    namespace = {'__name__': 'demo_geometry_test'}
    exec(compile(ast.Module(body=nodes[:stop], type_ignores=[]), 'd19_geometry', 'exec'), namespace)
    geo, angles = namespace['geo'], namespace['angles']
    # Sample the same physical detector extent for a lightweight rendering test.
    geo.nDetector = np.array([16, 20])
    geo.dDetector = geo.sDetector / geo.nDetector
    surface = surface_from_geometry(geo, angles)
    np.testing.assert_allclose(surface.pixel_centers_mm,
        np.broadcast_to(surface.pixel_centers_mm[0], surface.pixel_centers_mm.shape), atol=1e-4)
    assert np.ptp(surface.source_positions_mm[:, 1]) > 200
    animation = animate_geometry(geo, surface=surface)
    try:
        animation._fig.canvas.draw()
        for frame in (0, 4, 8):
            animation._func(frame)
            animation._fig.canvas.draw()
    finally:
        animation._draw_was_started = True
        plt.close(animation._fig)
