"""Small pass/fail harness shared by the stage checks."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class CheckList:
    def __init__(self, stage: int, location: str, variant: str | None = None, run: str | None = None):
        self.stage, self.location, self.variant, self.run_id = stage, location, variant, run
        self.results: list[dict] = []

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
            "stage": self.stage, "location": self.location, "passed": passed, "not_evaluated": skipped,
            **({"run": self.run_id} if self.run_id else {}),
            **({"variant": self.variant} if self.variant else {}),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checks": self.results,
        }
        stem = f"stage_{self.stage}" if isinstance(self.stage, int) else str(self.stage)
        # keyed by everything that identifies what was checked, so checking one location or run
        # never overwrites the report of another
        out = REPO_ROOT / "reports" / ("_".join([stem, self.location] + [p for p in (self.run_id, self.variant) if p]) + ".json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
        note = f" ({len(skipped)} not evaluated: {', '.join(skipped)})" if skipped else ""
        print(f"{stem.replace('_', ' ')}: {'PASSED' if passed else 'FAILED'}{note} -> {out}")
        return 0 if passed else 1
