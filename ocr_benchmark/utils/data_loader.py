import json
from pathlib import Path
from typing import Any, Dict, List


def load_local_data(folder: str) -> List[Dict[str, Any]]:
    """Load JSON files from a local folder.

    Each JSON file may contain either a single object or a list of objects.
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []

    files = sorted([f for f in p.glob("*.json") if f.is_file()])
    items: List[Dict[str, Any]] = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            parsed = json.loads(content)
            if isinstance(parsed, list):
                items.extend(parsed)
            else:
                items.append(parsed)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    return items
