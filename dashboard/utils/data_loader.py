import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, TypedDict

load_dotenv()


class BenchmarkRunMetadata(TypedDict):
    timestamp: str
    status: str
    run_by: str | None
    description: str | None
    total_documents: int | None
    created_at: str | None
    completed_at: str | None


def load_run_list_from_folder(
    results_dir: str = "results",
) -> List[BenchmarkRunMetadata]:
    """Load list of benchmark runs from the results directory"""
    results_path = Path(results_dir)
    result_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    runs = []

    for dir_path in result_dirs:
        timestamp = dir_path.name
        json_path = dir_path / "results.json"
        if json_path.exists():
            runs.append(
                {
                    "timestamp": timestamp,
                    "status": "completed",  # Assuming completed if file exists
                    "run_by": None,
                    "description": None,
                    "total_documents": None,
                    "created_at": format_timestamp(timestamp),
                    "completed_at": format_timestamp(timestamp),
                }
            )

    return sorted(runs, key=lambda x: x["timestamp"], reverse=True)


def load_results_for_run_from_folder(
    timestamp: str, results_dir: str = "results"
) -> Dict[str, Any]:
    """Load results for a specific run from folder"""
    results_path = Path(results_dir) / timestamp / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            # Assign id to each result if not already present
            for idx, result in enumerate(results):
                if "id" not in result:
                    result["id"] = idx
            total_documents = len(results)
            return {
                "results": results,
                "status": "completed",
                "run_by": None,
                "description": None,
                "total_documents": total_documents,
                "created_at": format_timestamp(timestamp),
                "completed_at": format_timestamp(timestamp),
            }
    return {}


def load_one_result_from_folder(
    timestamp: str, id: str, results_dir: str = "results"
) -> Dict[str, Any]:
    """Load one test case result from folder for a specific run and file"""
    results_path = Path(results_dir) / timestamp / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            for idx, result in enumerate(results):
                if idx == id:
                    return {
                        "result": result,
                        "status": "completed",
                        "run_by": None,
                        "description": None,
                        "created_at": format_timestamp(timestamp),
                        "completed_at": format_timestamp(timestamp),
                    }
    return {}


def load_run_list() -> List[BenchmarkRunMetadata]:
    """Load list of benchmark runs from either database or local files"""
    return load_run_list_from_folder()


def load_results_for_run(
    timestamp: str, include_metrics_only: bool = True
) -> Dict[str, Any]:
    """Load results for a specific run from either database or local files"""
    return load_results_for_run_from_folder(timestamp)


def load_one_result(timestamp: str, id: str) -> Dict[str, Any]:
    """Load one test case result from either database or local files"""
    return load_one_result_from_folder(timestamp, id)


def format_timestamp(timestamp: str) -> str:
    """Convert timestamp string to readable format"""
    return datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
