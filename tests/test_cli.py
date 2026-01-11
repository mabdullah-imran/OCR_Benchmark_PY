import subprocess
import sys


def test_cli_list_models():
    # Run the package CLI and check that it lists the sample models
    res = subprocess.run(
        [sys.executable, "-m", "ocr_benchmark", "--list-models"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    out = res.stdout + res.stderr
    # At least confirm the CLI printed the models header and at least one model entry
    assert "Loaded models:" in out
    models_lines = [l for l in out.splitlines() if l.strip() and l.strip().startswith("{")]
    assert len(models_lines) >= 1
