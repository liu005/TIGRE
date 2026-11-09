"""Run a curated CPU suite with native CUDA calls explicitly forbidden.

Stages source and test-only native import guards outside the checkout. This
does not build/install TIGRE or validate its CUDA binaries.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    "test_build_configuration.py",
    "test_sample_loader.py",
    "test_design_api.py",
    "test_headless_repository_boundary.py",
    "test_detector_geometry.py",
    "test_detector_geometry_remap.py",
    "test_geometry_yaml.py",
    "test_geometry_matlab_bridge.py",
    "test_operator_dense_oracles.py",
    "test_operator_pcsd.py",
]
NATIVE_IMPORTS = {
    "_Ax": ["_Ax_ext"],
    "_Atb": ["_Atb_ext"],
    "_AxSurface": ["_Ax_surface_ext", "_Atb_surface_ext", "_fdk_surface_ext"],
    "_gpuUtils": ["getGpuIdList", "getGpuNames"],
    "_RandomNumberGenerator": ["add_poisson", "add_noise"],
    "_tv_proximal": ["tvdenoise"],
    "_minTV": ["minTV"],
    "_AwminTV": ["AwminTV"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="tigre-cpu-") as temporary:
        stage = Path(temporary)
        source = stage / "Python"
        shutil.copytree(ROOT / "Python" / "tigre", source / "tigre",
                        ignore=shutil.ignore_patterns("*.pyd", "*.so", "*.dll", "__pycache__"))
        shutil.copytree(ROOT / "Common" / "data", stage / "Common" / "data")
        for module, names in NATIVE_IMPORTS.items():
            guard = (
                "def forbidden(*args, **kwargs):\n"
                "    raise AssertionError('Native CUDA call forbidden in CPU CI')\n"
            )
            guard += "\n".join(name + " = forbidden" for name in names) + "\n"
            (source / (module + ".py")).write_text(guard, encoding="utf-8")
        # Also inherited by subprocess-based API/boundary checks.
        (source / "sitecustomize.py").write_text(
            "import importlib.abc\nimport sys\n"
            "class NoGUI(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in {'tkinter', 'tigre_gui', 'tigre_geometry_design_tool'}:\n"
            "            raise AssertionError('GUI import forbidden in CPU CI: ' + fullname)\n"
            "sys.meta_path.insert(0, NoGUI())\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(source), MPLBACKEND="Agg")
        if os.name == "nt":
            # The existing Windows import initializer expects a CUDA bin path.
            (stage / "bin").mkdir()
            environment["CUDA_PATH"] = str(stage)
        preflight = """
import pathlib
import sys
import tigre
import _Ax
assert pathlib.Path(tigre.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
try:
    _Ax._Ax_ext()
except AssertionError as error:
    assert 'Native CUDA call forbidden' in str(error)
else:
    raise AssertionError('CPU native guard did not run')
try:
    import tkinter
except AssertionError as error:
    assert 'GUI import forbidden' in str(error)
else:
    raise AssertionError('CPU GUI guard did not run')
"""
        subprocess.run([sys.executable, "-c", preflight, str(source)],
                       cwd=stage, env=environment, check=True)
        command = [sys.executable, "-m", "pytest", "-q", "-rs"]
        if args.junitxml:
            command.append("--junitxml=" + str(args.junitxml.resolve()))
        command.extend(str(ROOT / "Python" / "tests" / name) for name in TESTS)
        print("CPU checks: staged source; GUI imports and native CUDA calls forbidden.", flush=True)
        return subprocess.run(command, cwd=stage, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())
