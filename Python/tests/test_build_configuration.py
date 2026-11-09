"""Exercise setup configuration without executing setup or requiring a compiler."""
import ast
import os
from pathlib import Path
import re
import subprocess
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_config():
    tree = ast.parse((ROOT / "Python/setup.py").read_text(encoding="utf-8"))
    names = {"get_cuda_version", "cuda_version_tuple", "compute_capability_args",
             "detect_gpu_compute_capabilities"}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in names or
             isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "CC_COMPATIBILITY_TABLE"
                 for target in node.targets)]
    namespace = dict(os=os, re=re, subprocess=subprocess, warnings=warnings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), namespace)
    return namespace


def test_mixed_devices_filter_both_native_and_ptx_targets(build_config):
    with pytest.warns(RuntimeWarning, match="120"):
        args = build_config["compute_capability_args"]("12.0", ["86", "120"])
    assert args == ["-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_86,code=compute_86"]


def test_all_detected_devices_incompatible_is_actionable(build_config):
    with pytest.warns(RuntimeWarning, match="120"):
        with pytest.raises(RuntimeError, match="Install a compatible CUDA toolkit"):
            build_config["compute_capability_args"]("11.8", ["120"])


@pytest.mark.parametrize("version,cap,compatible", [
    ("10.2", "30", True), ("11.0", "30", False),
    ("11.8", "35", True), ("11.8", "37", True),
    ("12.0", "35", False), ("12.0", "37", False),
    ("12.7", "100", False), ("12.8", "100", True),
    ("12.7", "120", False), ("12.8", "120", True),
    ("12.8", "61", True), ("12.9", "70", True),
    ("13.0", "61", False), ("13.0", "70", False),
    ("13.0", "75", True), ("11.8", "90", True),
])
def test_toolkit_architecture_boundaries(build_config, version, cap, compatible):
    select = build_config["compute_capability_args"]
    if compatible:
        assert "-gencode=arch=compute_{0},code=sm_{0}".format(cap) in select(version, [cap])
    else:
        with pytest.warns(RuntimeWarning):
            with pytest.raises(RuntimeError, match="No supported GPU compilation targets"):
                select(version, [cap])


def test_headless_build_targets_only_supported_architectures(build_config):
    args = build_config["compute_capability_args"]("13.0", [])
    assert "-gencode=arch=compute_75,code=sm_75" in args
    assert "-gencode=arch=compute_120,code=sm_120" in args
    assert not any("sm_61" in arg or "sm_70" in arg for arg in args)


def test_multi_digit_minor_version_selects_newer_targets(build_config):
    assert "-gencode=arch=compute_120,code=sm_120" in build_config[
        "compute_capability_args"]("12.10", ["120"])


def test_empty_gpu_query_uses_headless_fallback(build_config, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "")
    assert build_config["detect_gpu_compute_capabilities"]() == []


@pytest.mark.parametrize("version", ["11.8", "12.8", "12.10", "13.0"])
@pytest.mark.parametrize("version_file", [False, True])
def test_cuda_version_preserves_minor_component(build_config, monkeypatch, tmp_path,
                                               version, version_file):
    if version_file:
        (tmp_path / "version.txt").write_text("CUDA Version " + version + ".88\n")
    else:
        monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs:
                            ("Cuda compilation tools, release " + version + ", V" + version).encode())
    assert build_config["get_cuda_version"](str(tmp_path)) == version
    assert build_config["cuda_version_tuple"](version) == tuple(map(int, version.split(".")))


def test_unrecognized_cuda_version_has_actionable_error(build_config, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: b"unknown")
    with pytest.raises(RuntimeError, match="nvcc --version"):
        build_config["get_cuda_version"](str(tmp_path))


def test_package_declares_python38_minimum():
    tree = ast.parse((ROOT / "Python/setup.py").read_text(encoding="utf-8"))
    setup = next(node.value for node in tree.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name) and node.value.func.id == "setup")
    metadata = {keyword.arg: keyword.value for keyword in setup.keywords}
    assert ast.literal_eval(metadata["python_requires"]) == ">=3.8"


def test_python_sources_use_python38_grammar():
    for source in (ROOT / "Python/tigre").rglob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source), feature_version=(3, 8))


def test_geometry_document_annotations_support_python38():
    from typing import Optional, get_type_hints
    from tigre.utilities.geometry_io import GeometryDocument

    assert get_type_hints(GeometryDocument)["settings"] == Optional[dict]
