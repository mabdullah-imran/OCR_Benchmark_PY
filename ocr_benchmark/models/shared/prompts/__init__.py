from pathlib import Path

BASE = Path(__file__).parent

OCR_SYSTEM_PROMPT = (BASE / "ocr_system_prompt.txt").read_text(encoding="utf-8")
JSON_EXTRACTION_SYSTEM_PROMPT = (BASE / "json_extraction_system_prompt.txt").read_text(encoding="utf-8")
IMAGE_EXTRACTION_SYSTEM_PROMPT = (
    BASE / "image_extraction_system_prompt.txt"
).read_text(encoding="utf-8")

__all__ = [
    "OCR_SYSTEM_PROMPT",
    "JSON_EXTRACTION_SYSTEM_PROMPT",
    "IMAGE_EXTRACTION_SYSTEM_PROMPT",
]
