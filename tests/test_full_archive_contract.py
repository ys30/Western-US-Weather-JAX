from pathlib import Path

def test_full_archive_workflow_keeps_actual_model_and_data_bytes():
    text=(Path(__file__).resolve().parents[1]/".github/workflows/era5-archive-reproducibility.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "branches: [main]" in text
    assert "paths:" in text
    assert "era5-archive-reproducibility.yml" in text
    for fragment in ("input/*.nc","outputs/archive/**/*.npz","outputs/archive/**/*.json",
                     "resolved-requirements.txt","python-version.txt","git-commit.txt",
                     "sha256-manifest.json","--train-until 2021-01-01",
                     "--validate-until 2022-01-01","--test-start 2022-01-01"):
        assert fragment in text
