import os
import yaml
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import traceback
from tqdm import tqdm

# dotenv is optional; try to load environment variables if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

MODEL_CONCURRENCY: Dict[str, int] = {
    "aws-textract": 50,
    "azure-document-intelligence": 50,
    "claude-3-5-sonnet-20241022": 10,
    "gemini-2.0-flash-001": 30,
    "mistral-ocr": 5,
    "gpt-4o": 50,
    "qwen2.5-vl-32b-instruct": 10,
    "qwen2.5-vl-72b-instruct": 10,
    "google/gemma-3-27b-it": 10,
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": 10,
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": 10,
    "omniai": 30,
    "zerox": 50,
}

TIMEOUT_SECONDS = 10 * 60  # 10 minutes

# ---------------------------------------------------------------------------
# Helpers / best-effort imports from local modules (fallbacks if missing)
# ---------------------------------------------------------------------------


def _safe_import(name: str):
    try:
        module = __import__(name, fromlist=["*"])
        return module
    except Exception:
        return None


# evaluation helpers
_evaluation = _safe_import(".evaluation") if __package__ else None
try:
    from .evaluation import calculate_json_accuracy  # type: ignore
except Exception:

    def calculate_json_accuracy(true_json: Any, predicted_json: Any) -> Dict[str, Any]:
        return {
            "score": None,
            "jsonDiff": None,
            "fullJsonDiff": None,
            "jsonDiffStats": None,
        }


# model provider
try:
    from .models import get_model_provider  # type: ignore
except Exception:

    def get_model_provider(name: str):
        raise NotImplementedError(
            "get_model_provider is not implemented. Add model providers in the .models package."
        )


# utility functions
from .utils import (
    create_result_folder,
    load_local_data,
    write_to_file,
)  # type: ignore


# data / config
def load_models_config(models_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load models configuration from a YAML file.

    By default reads `models.yaml` from the repository root. This can be
    overridden by passing `models_path` or by setting the `MODELS_FILE`
    environment variable.
    """
    try:
        if models_path is None:
            models_path = os.getenv("MODELS_FILE") or str(
                Path(__file__).resolve().parents[1] / "models.yaml"
            )
        config_path = Path(models_path)
        if not config_path.exists():
            print(
                f"Models config not found at {config_path}; set MODELS_FILE env var or create a models.yaml"
            )
            return []
        contents = config_path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(contents) or {}
        return cfg.get("models", [])
    except Exception as e:
        print("Error loading models config:", e)
        return []


MODELS = load_models_config()
DATA_FOLDER = str(Path(__file__).resolve().parents[1] / "data")

# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def with_timeout(coro, operation: str):
    try:
        return await asyncio.wait_for(coro, timeout=TIMEOUT_SECONDS)
    except Exception as e:
        print(f"Timeout or error during {operation}: {e}")
        raise


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_benchmark():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_folder = create_result_folder(timestamp)

    data: List[Dict[str, Any]]
    data = load_local_data(DATA_FOLDER)

    results: List[Dict[str, Any]] = []

    # create progress bars (one per model configuration)
    progress_bars: Dict[str, Any] = {}
    for m in MODELS:
        key = (
            f"{m.get('extraction')} (IMG2JSON)"
            if m.get("directImageExtraction")
            else f"{m.get('ocr')} -> {m.get('extraction')}"
        )
        progress_bars[key] = tqdm(total=len(data), desc=key)

    async def process_with_model(m: Dict[str, Any]):
        ocr_model = m.get("ocr")
        extraction_model = m.get("extraction")
        direct_image = m.get("directImageExtraction", False)

        # concurrency for this model
        concurrency = min(
            MODEL_CONCURRENCY.get(ocr_model, 20),
            MODEL_CONCURRENCY.get(extraction_model, 20) if extraction_model else 20,
        )
        sem = asyncio.Semaphore(concurrency)

        provider_ocr = None
        provider_extraction = None
        try:
            provider_ocr = get_model_provider(ocr_model)
        except Exception:
            provider_ocr = None
        try:
            provider_extraction = (
                get_model_provider(extraction_model) if extraction_model else None
            )
        except Exception:
            provider_extraction = None

        async def process_item(item: Dict[str, Any]):
            async with sem:
                result: Dict[str, Any] = {
                    "fileUrl": item.get("imageUrl"),
                    "metadata": item.get("metadata"),
                    "jsonSchema": item.get("jsonSchema"),
                    "ocrModel": ocr_model,
                    "extractionModel": extraction_model,
                    "directImageExtraction": direct_image,
                    "trueMarkdown": item.get("trueMarkdownOutput"),
                    "trueJson": item.get("trueJsonOutput"),
                    "predictedMarkdown": None,
                    "predictedJson": None,
                    "jsonAccuracy": None,
                    "jsonDiff": None,
                    "fullJsonDiff": None,
                    "jsonDiffStats": None,
                    "jsonAccuracyResult": None,
                    "usage": None,
                    "error": None,
                }

                try:
                    if direct_image:
                        if provider_extraction is None:
                            raise RuntimeError(
                                "Extraction provider not available for direct image extraction"
                            )
                        extraction_result = await with_timeout(
                            provider_extraction.extract_from_image(
                                item.get("imageUrl"), item.get("jsonSchema")
                            ),
                            f"JSON extraction: {extraction_model}",
                        )
                        result["predictedJson"] = (
                            extraction_result.get("json")
                            if isinstance(extraction_result, dict)
                            else getattr(extraction_result, "json", None)
                        )
                        result["usage"] = {
                            "extraction": extraction_result.get("usage")
                            if isinstance(extraction_result, dict)
                            else getattr(extraction_result, "usage", None)
                        }
                    else:
                        ocr_result = None
                        if ocr_model == "ground-truth":
                            result["predictedMarkdown"] = item.get("trueMarkdownOutput")
                        else:
                            if provider_ocr is not None:
                                ocr_result = await with_timeout(
                                    provider_ocr.ocr(item.get("imageUrl")),
                                    f"OCR: {ocr_model}",
                                )
                                result["predictedMarkdown"] = (
                                    ocr_result.get("text")
                                    if isinstance(ocr_result, dict)
                                    else getattr(ocr_result, "text", None)
                                )
                                result["usage"] = {
                                    "ocr": ocr_result.get("usage")
                                    if isinstance(ocr_result, dict)
                                    else getattr(ocr_result, "usage", None)
                                }

                        if provider_extraction is not None:
                            extraction_result = await with_timeout(
                                provider_extraction.extract_from_text(
                                    result.get("predictedMarkdown"),
                                    item.get("jsonSchema"),
                                    getattr(ocr_result, "imageBase64s", None)
                                    if ocr_result
                                    else None,
                                ),
                                f"JSON extraction: {extraction_model}",
                            )
                            result["predictedJson"] = (
                                extraction_result.get("json")
                                if isinstance(extraction_result, dict)
                                else getattr(extraction_result, "json", None)
                            )
                            # best-effort merge usage
                            res_usage = {}
                            if isinstance(result.get("usage"), dict):
                                res_usage.update(result.get("usage"))
                            if isinstance(
                                extraction_result, dict
                            ) and extraction_result.get("usage"):
                                res_usage.update(
                                    {"extraction": extraction_result.get("usage")}
                                )
                            result["usage"] = res_usage

                    if result.get("predictedJson"):
                        json_accuracy_result = calculate_json_accuracy(
                            item.get("trueJsonOutput"), result.get("predictedJson")
                        )
                        if isinstance(json_accuracy_result, dict):
                            result["jsonAccuracy"] = json_accuracy_result.get("score")
                            result["jsonDiff"] = json_accuracy_result.get("jsonDiff")
                            result["fullJsonDiff"] = json_accuracy_result.get(
                                "fullJsonDiff"
                            )
                            result["jsonDiffStats"] = json_accuracy_result.get(
                                "jsonDiffStats"
                            )
                            result["jsonAccuracyResult"] = json_accuracy_result

                except Exception as e:
                    result["error"] = str(e)
                    print(
                        f"Error processing {item.get('imageUrl')} with {ocr_model} and {extraction_model}: \n",
                        traceback.format_exc(),
                    )

                # update progress bar
                key = (
                    f"{extraction_model} (IMG2JSON)"
                    if direct_image
                    else f"{ocr_model}-{extraction_model}"
                )
                pbar = progress_bars.get(key)
                if pbar:
                    try:
                        pbar.update(1)
                    except Exception:
                        pass

                return result

        # launch all item tasks for this model
        tasks = [asyncio.create_task(process_item(item)) for item in data]
        model_results = await asyncio.gather(*tasks)
        results.extend(model_results)

    # launch model-level tasks in parallel
    await asyncio.gather(*(process_with_model(m) for m in MODELS))

    # close progress bars
    for p in progress_bars.values():
        try:
            p.close()
        except Exception:
            pass

    write_to_file(str(result_folder / "results.json"), results)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
