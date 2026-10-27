import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("_AxSurface")

from test_surface_algorithms import SURFACES, volume_geo
from test_u_detector import u_surface
from tigre.utilities.visualization.animate_geometry import animate_geometry, _detector_runs


@pytest.mark.parametrize("name", ["arc", "modular_gaps", "u"])
def test_animate_geometry_with_surface(name):
    surf = u_surface() if name == "u" else SURFACES[name]()
    ax = animate_geometry(volume_geo(), surface=surf, pos=3, animate=False, view=(90, 0))
    assert (ax.elev, ax.azim) == (90, 0)
    ax = animate_geometry(volume_geo(), surface=surf, pos=3, animate=False)
    assert "view 3/" in ax.get_title()
    n_modules = {"arc": 1, "modular_gaps": 3, "u": 3}[name]
    assert len(_detector_runs(surf)) == n_modules
    ani = animate_geometry(volume_geo(), surface=surf, animate=True)
    ani._fig.canvas.draw()
    ani._func(2)  # one update step
    ani._draw_was_started = True


def test_surface_mode_rejects_object_rotation():
    with pytest.raises(ValueError):
        animate_geometry(volume_geo(), surface=SURFACES["arc"](), rotation="obj")


def test_module_slices_define_runs():
    surf = SURFACES["arc"]()
    n = surf.valid_mask.shape[2] // 2
    split = type(surf)(surf.source_positions_mm, surf.pixel_centers_mm, surf.valid_mask,
                       module_slices=[(slice(0, 1), slice(0, n)), (slice(0, 1), slice(n, 2 * n))])
    assert _detector_runs(split) == [(0, n), (n, 2 * n)]
    assert split.subset([0, 1])._module_slices == split._module_slices
    with pytest.raises(ValueError, match="inside"):
        type(surf)(surf.source_positions_mm, surf.pixel_centers_mm,
                   module_slices=[(slice(0, 1), slice(0, 10 * n))])


def test_surface_mode_updates_per_view_masks():
    surf = SURFACES["arc"]()
    mask = surf.valid_mask.copy()
    mask[1, :, :5] = False
    varying = type(surf)(surf.source_positions_mm, surf.pixel_centers_mm, mask)
    ani = animate_geometry(volume_geo(), surface=varying, animate=True)
    ani._func(1)
    assert f"{int(mask[1].sum())} valid pixels" in ani._fig.axes[0].get_title()
    ani._fig.canvas.draw()
    ani._draw_was_started = True
