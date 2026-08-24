"""Which algorithms actually honour mu_max / support, and which cannot.

`apply_constraints()` projects the iterate onto the feasible set
(nonnegativity, an attenuation ceiling, a known-air support). The base class
calls it from its init paths and from art_data_minimizing's block loop - so an
algorithm that overrides run_main_iter AND does not route through
dataminimizing gets the ceiling applied to its warm start and never again.

That was silently true of MLEM, which is the case that matters: an attenuation
ceiling was expected to tame its long-standing hot voxels and never could,
because it never reached the iterations. Measured before the fix, mu_max=0.5
on a 64^3 phantom: MLEM returned 3.46 while SIRT/OS_SART/FAST_OS_SART/ASD_POCS
all returned exactly 0.5.

The Krylov family is deliberately NOT in the honouring list. Projecting a CGLS
iterate mid-run would break the conjugacy its recursions assume - r, p and
gamma are updated recursively against the UNPROJECTED step - so a constraint
there has to be a post-hoc clip of the returned volume, not a per-iteration
projection. Callers passing mu_max to a Krylov method should know it binds on
the warm start only.
"""
import numpy as np
import pytest

import tigre
from tigre.algorithms import mlem, sirt, os_sart, fast_os_sart, fista, asd_pocs, cgls

MU = 0.5


@pytest.fixture(scope="module")
def noisy_problem():
    geo = tigre.geometry(mode="cone", nVoxel=np.array([64, 64, 64]), default=True)
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False, dtype=np.float32)
    z, y, x = np.mgrid[:64, :64, :64].astype(np.float32)
    ph = np.zeros((64, 64, 64), dtype=np.float32)
    ph[((x - 30) ** 2 + (y - 32) ** 2 + (z - 32) ** 2) < 14 ** 2] = 1.0
    proj = tigre.Ax(ph, geo, angles, "Siddon")
    rng = np.random.default_rng(0)
    noisy = proj + rng.standard_normal(proj.shape, dtype=np.float32) * np.float32(0.05 * proj.max())
    return geo, angles, ph, noisy


@pytest.mark.parametrize("alg", [mlem, sirt, os_sart, fast_os_sart, fista, asd_pocs])
def test_mu_max_binds(noisy_problem, alg):
    geo, angles, ph, noisy = noisy_problem
    res = alg(noisy.copy(), geo, angles, niter=20, mu_max=MU, verbose=False)
    assert float(res.max()) <= MU + 1e-4, (
        f"{alg.__name__} returned {float(res.max()):.4f} against a ceiling of {MU}")


@pytest.mark.parametrize("alg", [mlem, sirt, os_sart, fast_os_sart, fista])
def test_nonnegativity_holds_for_algorithms_that_project_last(noisy_problem, alg):
    """ASD_POCS is excluded on purpose, not overlooked.

    Its TV step runs AFTER the projection - `res = minimizeTV(res, dtvg)` is
    the last thing each iteration does - so minTV can leave small negatives
    behind: measured -1.3e-4 against values around 0.5. The ceiling still
    binds, which is what the prior is for. Re-projecting after the TV step
    would be defensible (POCS is named for projection onto convex sets) but it
    would change published POCS results for a cosmetic gain, so it is recorded
    here rather than done quietly.
    """
    geo, angles, ph, noisy = noisy_problem
    res = alg(noisy.copy(), geo, angles, niter=20, mu_max=MU, verbose=False)
    assert float(res.min()) >= -1e-6


def test_mlem_ceiling_removes_its_hot_voxels(noisy_problem):
    """The point of the exercise, not just the mechanism."""
    geo, angles, ph, noisy = noisy_problem
    free = mlem(noisy.copy(), geo, angles, niter=50, verbose=False)
    capped = mlem(noisy.copy(), geo, angles, niter=50, mu_max=1.2, verbose=False)
    assert int((free > 5 * ph.max()).sum()) > 0, "phantom no longer provokes hot voxels"
    assert int((capped > 5 * ph.max()).sum()) == 0


def test_mlem_keeps_nonnegativity_even_if_asked_not_to(noisy_problem):
    """MLEM's update is multiplicative: one negative factor flips a voxel's
    sign for good. Its old hand-rolled clip was unconditional and routing
    through apply_constraints must not turn that into a user option."""
    geo, angles, ph, noisy = noisy_problem
    res = mlem(noisy.copy(), geo, angles, niter=20, noneg=False, verbose=False)
    assert float(res.min()) >= 0.0


def test_krylov_ceiling_is_warm_start_only(noisy_problem):
    """Documents the limitation rather than pretending it is not there."""
    geo, angles, ph, noisy = noisy_problem
    res = cgls(noisy.copy(), geo, angles, niter=20, mu_max=MU, init="FDK", verbose=False)
    assert np.isfinite(res).all()
    if float(res.max()) <= MU + 1e-4:
        pytest.skip("CGLS happened to stay under the ceiling; nothing proven either way")
