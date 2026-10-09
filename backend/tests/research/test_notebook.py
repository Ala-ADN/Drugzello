"""Keep the tutorial executable without requiring a Jupyter server in CI."""

import json
from pathlib import Path


def test_default_notebook_executes_shared_pipeline(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    notebook = json.loads((root / "notebooks/megan_pytorch.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            exec(compile("".join(cell["source"]), f"megan_pytorch.ipynb:cell-{i}", "exec"), namespace)
    assert namespace["RUN_ESOL"] is False
    assert namespace["summary"]["fold_count"] == 3
