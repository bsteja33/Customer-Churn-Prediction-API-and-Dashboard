"""tests/test_integration_train.py"""

"""Integration test for the training pipeline.

- Takes a small 200‑row sample of the original CSV.
- Writes a temporary ``config.yaml`` that points to the sample and writes the model to a temporary location.
- Executes ``src/train.py`` via a subprocess.
- Verifies that the model artifact was created.
"""
import pathlib
import subprocess
import sys
import yaml
import pandas as pd


def test_training_on_small_subset(tmp_path: pathlib.Path):
    """Run ``train.py`` on a 200‑row slice and check the model file.

    The test creates a temporary data file and a temporary config that
    redirects both the input CSV and the model output path.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    # ------------------------------------------------------------------
    # 1. Create a 200‑row sample CSV in the temporary directory
    # ------------------------------------------------------------------
    original_csv = repo_root / "data" / "telecom_customer_churn.csv"
    sample_csv = tmp_path / "sample.csv"
    # Read only the first 200 rows (skipheader already included)
    df_sample = pd.read_csv(original_csv, nrows=200)
    df_sample.to_csv(sample_csv, index=False)

    # ------------------------------------------------------------------
    # 2. Load the original config and patch paths
    # ------------------------------------------------------------------
    original_cfg_path = repo_root / "config.yaml"
    cfg = yaml.safe_load(original_cfg_path.read_text())
    # Point the data loader to the tiny sample
    cfg["data"]["raw_path"] = str(sample_csv)
    # Write model artifact to a temporary location
    tmp_model_path = tmp_path / "tmp_model.pkl"
    cfg["model"]["save_path"] = str(tmp_model_path)

    # Write the temporary config file
    temp_cfg_path = tmp_path / "test_config.yaml"
    temp_cfg_path.write_text(yaml.safe_dump(cfg))

    # ------------------------------------------------------------------
    # 3. Run the training script via subprocess
    # ------------------------------------------------------------------
    # Use the repository root as the working directory so relative imports work
    result = subprocess.run(
        [sys.executable, "src/train.py", "--config", str(temp_cfg_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )

    # If the script failed, surface stdout/stderr for debugging
    assert result.returncode == 0, f"train.py failed: {result.stdout}\n{result.stderr}"

    # ------------------------------------------------------------------
    # 4. Verify the model artifact was written
    # ------------------------------------------------------------------
    assert tmp_model_path.exists(), "Model artifact was not created"
