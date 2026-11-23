"""Test the submission writer and validator in src/common.py.

Writes submissions/constant_0.5.csv (a valid file that scores AUC 0.5), then
checks that the validator rejects each kind of malformed file.
Run: .venv/bin/python scripts/00_test_submission.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import list_test_ids, validate_submission, write_submission  # noqa: E402

ids = list_test_ids()
path = write_submission({i: 0.5 for i in ids}, "constant_0.5")
print("valid file written:", path.name)

good = pd.read_csv(path, dtype={"index": str})
bad_cases = {
    "wrong header": lambda d: d.rename(columns={"score": "label"}),
    "unsorted": lambda d: d.iloc[::-1],
    "missing row": lambda d: d.iloc[:-1],
    "duplicate row": lambda d: pd.concat([d, d.iloc[:1]]),
    "score > 1": lambda d: d.assign(score=d["score"] + 0.6),
    "NaN score": lambda d: d.assign(score=d["score"].where(d.index > 0)),
}
with tempfile.TemporaryDirectory() as tmp:
    for name, corrupt in bad_cases.items():
        p = Path(tmp) / "bad.csv"
        corrupt(good.copy()).to_csv(p, index=False)
        try:
            validate_submission(p)
        except AssertionError as e:
            print(f"  rejected ({name}): {e}")
        else:
            raise SystemExit(f"FAILED: validator accepted a file with {name}")
print("ALL SUBMISSION TESTS PASSED")
