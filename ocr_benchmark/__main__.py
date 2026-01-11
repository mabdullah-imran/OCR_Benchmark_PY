import argparse
import asyncio
import importlib
import sys
from typing import Optional

from . import load_models_config, run_benchmark


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_benchmark", description="Run OCR benchmark"
    )
    parser.add_argument(
        "--models-file", help="Path to models YAML file (overrides MODELS_FILE env var)"
    )
    parser.add_argument("--data-dir", help="Path to data folder (overrides default)")
    parser.add_argument(
        "--list-models", action="store_true", help="List loaded models and exit"
    )
    args = parser.parse_args(argv)

    pkg = importlib.import_module("ocr_benchmark")

    # Reload models if requested
    if args.models_file:
        pkg.MODELS = load_models_config(args.models_file)
    else:
        # ensure MODELS is populated from default location
        pkg.MODELS = load_models_config()

    if args.data_dir:
        pkg.DATA_FOLDER = args.data_dir

    if args.list_models:
        print("Loaded models:")
        for m in pkg.MODELS:
            print(m)
        return 0

    if not pkg.MODELS:
        print(
            "No models configured. Create a models.yaml in the repo root or pass --models-file."
        )
        return 2

    try:
        asyncio.run(run_benchmark())
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130
    except Exception as e:
        print("Benchmark failed:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
