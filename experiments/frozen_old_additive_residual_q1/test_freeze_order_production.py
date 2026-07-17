from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent.parent
DATASET_SHA256 = "4f10cfcf545c97eb3f56e8511921a1a6b50fa5ab166fac2eb3575eacf84b71ba"


def main() -> None:
    julia = shutil.which("julia")
    assert julia is not None, "Julia is not available on PATH"
    eligible_rows = list(range(1, 2125))
    eligibility = {
        "status": "eligibility_complete",
        "dataset_sha256": DATASET_SHA256,
        "training_episode_ids": list(range(1, 13)),
        "training_eligible_count": 2124,
        "training_eligible_rows": eligible_rows,
        "offline_rows_loaded": False,
    }
    with tempfile.TemporaryDirectory(prefix="q1-freeze-order-production-") as temporary:
        temporary_root = Path(temporary)
        eligibility_path = temporary_root / "synthetic_eligibility.json"
        output_path = temporary_root / "frozen_order.json"
        eligibility_path.write_text(json.dumps(eligibility), encoding="utf-8")

        subprocess.run(
            [
                julia,
                "--startup-file=no",
                "--history-file=no",
                f"--project={REPOSITORY}",
                str(ROOT / "freeze_order.jl"),
                str(eligibility_path),
                str(output_path),
            ],
            check=True,
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
        )

        assert output_path.is_file(), "production argv branch did not write its output"
        result = json.loads(output_path.read_text(encoding="utf-8"))
        ordered_rows = result["ordered_rows"]
        expected_digest = hashlib.sha256(
            ",".join(str(row) for row in ordered_rows).encode("utf-8")
        ).hexdigest()
        assert result["status"] == "q1_order_frozen"
        assert result["update_count"] == 2000
        assert len(ordered_rows) == 8000
        assert len(result["minibatches"]) == 2000
        assert result["ordered_rows_sha256"] == expected_digest
        assert dt.datetime.fromisoformat(result["generated_at"])
        assert result["eligibility_path"] == str(eligibility_path.resolve())
        assert result["offline_rows_loaded"] is False
        assert result["validation_or_test_seed_loaded"] is False
        assert result["game_seed_loaded"] is False

    print("Q1 freeze-order production argv synthetic check passed")


if __name__ == "__main__":
    main()
