"""Run this to forcibly clear the cache DB."""
import tempfile, os
from pathlib import Path

db = Path(tempfile.gettempdir()) / "finance_scenario_cache.db"
for suffix in ["", "-wal", "-shm"]:
    f = Path(str(db) + suffix)
    if f.exists():
        try:
            f.unlink()
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

print("Done. You can now restart scenario_agent.py")