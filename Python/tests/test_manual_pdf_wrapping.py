"""Verify PDF filters provide break opportunities for long inline code."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_inline_code_wrapping_filter():
    pandoc = shutil.which("pandoc")
    if not pandoc:
        pytest.skip("Pandoc is required for manual generation")
    filter_path = Path(__file__).resolve().parents[2] / "Frontispiece" / "manual-tables.lua"
    source = (
        "`test_geometry_matlab_bridge.py` "
        "[`d_detector_surface_gct.py`](demo.py) "
        "`.\\Frontispiece\\Export_User_Manual.ps1`\n\n"
        "```python\nlong_test_geometry_matlab_bridge.py\n```\n"
    )

    def convert(output_format):
        return subprocess.run(
            [pandoc, "--from=gfm", f"--to={output_format}", f"--lua-filter={filter_path}"],
            input=source, text=True, capture_output=True, check=True,
        ).stdout

    latex = convert("latex")
    assert r"\texttt{test\_}\allowbreak{}" in latex
    assert r"\href{demo.py}{" in latex
    assert r"\texttt{d\_}\allowbreak{}" in latex
    assert r"\textbackslash{}" in latex
    block = latex.split(r"\begin{Highlighting}", 1)[1]
    assert r"long\_test\_geometry\_matlab\_bridge.py" in block
    assert r"\allowbreak{}" not in block
    html = convert("html")
    assert "<code>test_geometry_matlab_bridge.py</code>" in html
    assert "<code>d_detector_surface_gct.py</code>" in html
