from pathlib import Path
import os
import shutil
import subprocess

def test_spooled_script_uses_submitted_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spool = tmp_path / "slurm_spool"
    spool.mkdir()
    script = spool / "slurm_script"
    shutil.copyfile(root / "scripts/slurm_worker.sh", script)
    stub = tmp_path / "python_stub"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$PWD" "$PYTHONPATH" "$@"\n')
    stub.chmod(0o755)
    result = subprocess.run(["bash", str(script), "manifest.json", str(stub)],
                            cwd=root, capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert lines[0] == str(root)
    assert lines[1].split(":")[0] == str(root)
    assert lines[2:] == ["-m", "hipe.experiments", "worker", "--manifest", "manifest.json"]
