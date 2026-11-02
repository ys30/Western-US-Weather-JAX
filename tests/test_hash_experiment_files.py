import hashlib
import json
from scripts.hash_experiment_files import manifest


def test_hash_files_and_skip_manifest_itself(tmp_path):
    root=tmp_path / "inputs"
    root.mkdir()
    data=root/"example.nc"
    data.write_bytes(b"test-era5-bytes")
    dest=root/"manifest.json"
    dest.write_text("{}")
    result=manifest([str(root)],str(dest))
    assert result["count"]==1
    assert result["files"][0]["sha256"]==hashlib.sha256(b"test-era5-bytes").hexdigest()
    assert result["files"][0]["bytes"]==15


def test_missing_root_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        manifest([str(tmp_path/"missing")],str(tmp_path/"out.json"))
