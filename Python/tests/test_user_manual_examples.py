"""Execute the Python workflows in the user manual against the current APIs."""

import re
from pathlib import Path


def test_python_manual_workflows(tmp_path, monkeypatch):
    manual = Path(__file__).resolve().parents[2] / "Frontispiece" / "TIGRE_User_Manual.md"
    text = manual.read_text(encoding="utf-8")
    blocks = dict(
        re.findall(r"```python\n# manual-example: (python-[\w-]+)\n(.*?)```", text, re.DOTALL)
    )
    names = (
        "python-conventional",
        "python-surface",
        "python-operator",
        "python-pocs",
        "python-interchange",
    )
    assert set(blocks) == set(names)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for name in names:
        exec(compile(blocks[name], str(manual), "exec"), namespace)
