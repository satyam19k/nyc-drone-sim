"""Small pass/fail harness shared by the checks."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# Reports and pictures go to docs/validation, or to NYC_DRONE_VALIDATION_DIR when it is set
# (the nyc-drone command sets it to the run's own folder).
VALIDATION = Path(os.environ.get("NYC_DRONE_VALIDATION_DIR") or REPO_ROOT / "docs" / "validation")
IMAGES, REPORTS = VALIDATION / "images", VALIDATION / "reports"


def shown(p: Path) -> str:
    """A path for a report: relative to the repository when inside it, else as it is (a run's
    own folder given with --out can be anywhere)."""
    p = Path(p)
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


class CheckList:
    def __init__(self, name: str, location: str, variant: str | None = None, run: str | None = None):
        self.name, self.location, self.variant, self.run_id = name, location, variant, run
        self.results: list[dict] = []
        IMAGES.mkdir(parents=True, exist_ok=True)

    def check(self, gid: str, ok: bool, detail, warn: bool = False):
        status = "PASS" if ok else "FAIL"
        if ok and warn:
            status = "WARN"
        self.results.append({"id": gid, "status": status, "detail": detail})
        print(f"[{status}] {gid}: {detail}")

    def run(self, gid: str, fn):
        """fn returns (ok, detail) or (ok, detail, warn). Exceptions are failures.

        ok = None means the check could not be made: it is recorded as NOT EVALUATED, which is
        neither a pass nor a failure, and is counted in the report.
        """
        try:
            out = fn()
            if out[0] is None:
                self.results.append({"id": gid, "status": "NOT EVALUATED", "detail": out[1]})
                print(f"[NOT EVALUATED] {gid}: {out[1]}")
                return
            self.check(gid, bool(out[0]), out[1], warn=len(out) > 2 and bool(out[2]))
        except Exception as e:  # a crashed check is a failed check
            self.check(gid, False, f"exception: {type(e).__name__}: {e}")

    def finish(self) -> int:
        passed = all(r["status"] != "FAIL" for r in self.results)
        skipped = [r["id"] for r in self.results if r["status"] == "NOT EVALUATED"]
        rep = {
            "check": self.name, "location": self.location, "passed": passed, "not_evaluated": skipped,
            **({"run": self.run_id} if self.run_id else {}),
            **({"variant": self.variant} if self.variant else {}),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checks": self.results,
        }
        stem = str(self.name)
        # keyed by everything that identifies what was checked, so checking one location or run
        # never overwrites the report of another
        out = REPORTS / ("_".join([stem, self.location] + [p for p in (self.run_id, self.variant) if p]) + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
        note = f" ({len(skipped)} not evaluated: {', '.join(skipped)})" if skipped else ""
        print(f"{stem} checks: {'PASSED' if passed else 'FAILED'}{note} -> {out}")
        return 0 if passed else 1
