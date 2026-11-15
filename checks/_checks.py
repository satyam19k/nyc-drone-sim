"""Small pass/fail harness shared by the stage checks."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class CheckList:
    def __init__(self, stage: int, location: str, variant: str | None = None):
        self.stage, self.location, self.variant = stage, location, variant
        self.results: list[dict] = []

    def check(self, gid: str, ok: bool, detail, warn: bool = False):
        status = "PASS" if ok else "FAIL"
        if ok and warn:
            status = "WARN"
        self.results.append({"id": gid, "status": status, "detail": detail})
        print(f"[{status}] {gid}: {detail}")

    def run(self, gid: str, fn):
        """fn returns (ok, detail) or (ok, detail, warn). Exceptions are failures."""
        try:
            out = fn()
            self.check(gid, bool(out[0]), out[1], warn=len(out) > 2 and bool(out[2]))
        except Exception as e:  # a crashed check is a failed check
            self.check(gid, False, f"exception: {type(e).__name__}: {e}")

    def finish(self) -> int:
        passed = all(r["status"] != "FAIL" for r in self.results)
        rep = {
            "stage": self.stage, "location": self.location, "passed": passed,
            **({"variant": self.variant} if self.variant else {}),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checks": self.results,
        }
        stem = f"stage_{self.stage}" if isinstance(self.stage, int) else str(self.stage)
        out = REPO_ROOT / "reports" / f"{stem}{'_' + self.variant if self.variant else ''}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
        print(f"{stem.replace('_', ' ')}: {'PASSED' if passed else 'FAILED'} -> {out}")
        return 0 if passed else 1
