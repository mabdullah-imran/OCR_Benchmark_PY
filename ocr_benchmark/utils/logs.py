from pathlib import Path
from typing import Any
import json


def create_result_folder(folder_name: str) -> Path:
    """Create a results folder at the repository's `results/<folder_name>` path and return it."""
    results_folder = Path(__file__).resolve().parents[1] / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    folder_path = results_folder / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def write_to_file(file_path: str, content: Any) -> None:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(content, default=str, indent=2), encoding="utf-8")


def write_result_to_file(output_dir: str, file_name: str, result: Any) -> None:
    output_path = Path(output_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")


__all__ = ["create_result_folder", "write_to_file", "write_result_to_file"]
