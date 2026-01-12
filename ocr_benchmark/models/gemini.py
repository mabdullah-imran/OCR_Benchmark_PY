import os
import time
import json
from typing import Any, Dict, Optional
from pathlib import Path
import traceback
import requests
import asyncio

from google import genai
from google.genai import types

from .base import BaseModel
from .shared import (
    JSON_EXTRACTION_SYSTEM_PROMPT,
    OCR_SYSTEM_PROMPT,
    IMAGE_EXTRACTION_SYSTEM_PROMPT,
)
from .shared.token_cost import calculate_token_cost
from ..utils import get_mime_type
from ..types.model import ExtractionResult


api_key = os.getenv("GOOGLE_AI_API_KEY")
_client = None
if api_key:
    try:
        _client = genai.Client(api_key=api_key)
    except Exception:
        _client = None


# Helper: extract token counts from various response shapes produced by GenAI SDKs
def _extract_token_counts(response: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) as ints (defaults to 0).

    The GenAI Python SDK has varied response shapes; try a few common
    locations (response.usageMetadata, response.usage, response.metadata).
    """
    meta = response.usage_metadata
    input_tokens = getattr(meta, "promptTokenCount", None) or getattr(
        meta, "prompt_token_count", None
    )
    output_tokens = getattr(meta, "candidatesTokenCount", None) or getattr(
        meta, "candidates_token_count", None
    )

    try:
        in_t = int(input_tokens) if input_tokens else 0
    except Exception:
        in_t = 0
    try:
        out_t = int(output_tokens) if output_tokens else 0
    except Exception:
        out_t = 0

    return in_t, out_t


class Gemini(BaseModel):
    """Gemini model provider using Google GenAI Python SDK (best-effort).

    Notes:
    - The Google GenAI Python SDK may expose slightly different APIs across
      versions. This implementation uses common patterns but is defensive
      and will raise helpful errors if the client isn't configured.
    """

    def __init__(self, model: str, output_dir: Optional[str] = None):
        super().__init__(model, output_dir)
        if _client is None:
            raise RuntimeError(
                "Google GenAI client is not configured. Set GENAI_API_KEY env var."
            )
        self.client = _client
        self.config = types.GenerateContentConfig(
            max_output_tokens=65535,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ],
        )

    async def ocr(self, image_path: str) -> ExtractionResult:
        start = time.perf_counter()

        # fetch image bytes in a thread to avoid blocking the event loop
        def _fetch_bytes(path: str):
            if Path(path).exists():
                return Path(path).read_bytes()
            resp = requests.get(path, timeout=60)
            resp.raise_for_status()
            return resp.content

        try:
            data = await asyncio.to_thread(_fetch_bytes, image_path)

            # Create an image input based on SDK's types when available
            image_part = types.Part.from_bytes(
                data=data, mime_type=get_mime_type(image_path)
            )

            self.config.system_instruction = OCR_SYSTEM_PROMPT
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.model, config=self.config, contents=[image_part]
                )
            )

            # Extract text from known response shapes
            text = response.text if response.text else str(response)
            usage = dict(response.usage_metadata) if response.usage_metadata else {}

            end = time.perf_counter()

            # Extract token counts and compute costs
            in_t, out_t = _extract_token_counts(response)
            try:
                input_cost = calculate_token_cost(self.model, "input", in_t)
                output_cost = calculate_token_cost(self.model, "output", out_t)
                total_cost = (input_cost or 0) + (output_cost or 0)
            except Exception:
                input_cost = output_cost = total_cost = None

            usage_payload = {
                "duration": (end - start),
                "inputTokens": in_t,
                "outputTokens": out_t,
                "totalTokens": in_t + out_t,
                "inputCost": input_cost,
                "outputCost": output_cost,
                "totalCost": total_cost,
            }

            merged_usage = {**usage, **usage_payload}

            return {
                "text": text,
                "usage": merged_usage, # pyright: ignore[reportReturnType]
            }
        except Exception as e:
            print("Gemini OCR error:", e)
            print(traceback.format_exc())
            raise

    async def extract_from_text(
        self, text: Any, schema: Dict[str, Any], imageBase64s=None
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        filtered_schema = self.convert_schema_for_gemini(schema)

        self.config.response_schema = filtered_schema
        self.config.system_instruction = JSON_EXTRACTION_SYSTEM_PROMPT
        try:
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[text],
                    config=self.config,
                )
            )

            jtext = response.text if response.text else str(response)

            json_obj = json.loads(jtext)
            end = time.perf_counter()

            usage = dict(response.usage_metadata) if response.usage_metadata else {}

            duration = end - t0

            in_t, out_t = _extract_token_counts(response)
            try:
                input_cost = calculate_token_cost(self.model, "input", in_t)
                output_cost = calculate_token_cost(self.model, "output", out_t)
                total_cost = (input_cost or 0) + (output_cost or 0)
            except Exception:
                input_cost = output_cost = total_cost = None

            usage_payload = {
                "duration": duration,
                "inputTokens": in_t,
                "outputTokens": out_t,
                "totalTokens": in_t + out_t,
                "inputCost": input_cost,
                "outputCost": output_cost,
                "totalCost": total_cost,
            }

            merged_usage = {**usage, **usage_payload}

            return {"json": json_obj, "usage": merged_usage} # pyright: ignore[reportReturnType]
        except Exception as e:
            print("Gemini extract_from_text error:", e)
            print(traceback.format_exc())
            raise

    async def extract_from_image(
        self, image_path: str, schema: Dict[str, Any]
    ) -> ExtractionResult:
        t0 = time.perf_counter()

        def _fetch_bytes(path: str):
            if Path(path).exists():
                return Path(path).read_bytes()
            resp = requests.get(path, timeout=60)
            resp.raise_for_status()
            return resp.content

        try:
            data = await asyncio.to_thread(_fetch_bytes, image_path)

            image_part = types.Part.from_bytes(
                data=data, mime_type=get_mime_type(image_path)
            )

            filtered_schema = self.convert_schema_for_gemini(schema)

            self.config.response_schema = filtered_schema
            self.config.system_instruction = IMAGE_EXTRACTION_SYSTEM_PROMPT
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=[image_part],
                    config=self.config,
                )
            )

            jtext = response.text if response.text else str(response)

            json_obj = json.loads(jtext)
            end = time.perf_counter()

            usage = dict(response.usage_metadata) if response.usage_metadata else {}

            duration = end - t0

            in_t, out_t = _extract_token_counts(response)
            try:
                input_cost = calculate_token_cost(self.model, "input", in_t)
                output_cost = calculate_token_cost(self.model, "output", out_t)
                total_cost = (input_cost or 0) + (output_cost or 0)
            except Exception:
                input_cost = output_cost = total_cost = None

            usage_payload = {
                "duration": duration,
                "inputTokens": in_t,
                "outputTokens": out_t,
                "totalTokens": in_t + out_t,
                "inputCost": input_cost,
                "outputCost": output_cost,
                "totalCost": total_cost,
            }

            merged_usage = {**usage, **usage_payload}

            return {"json": json_obj, "usage": merged_usage} # pyright: ignore[reportReturnType]
        except Exception as e:
            print("Gemini extract_from_image error:", e)
            print(traceback.format_exc())
            raise

    def convert_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        new_schema = json.loads(json.dumps(schema))

        def process_node(node: Any):
            if not node or not isinstance(node, dict):
                return node

            if node.get("type") == "enum" and node.get("enum") is not None:
                node["type"] = "string"
            if "enum" in node and not node.get("type"):
                node["type"] = "string"

            if "additionalProperties" in node:
                node.pop("additionalProperties", None)

            if "not" in node:
                not_node = node.get("not")
                if isinstance(not_node, dict) and not_node.get("type") == "null":
                    node.pop("not", None)
                    node["nullable"] = False
                else:
                    process_node(not_node)

            if node.get("type") == "array" and node.get("items"):
                items = node["items"]
                if node.get("required"):
                    if not items.get("required"):
                        items["required"] = node["required"]
                    else:
                        items["required"] = list(
                            {*items.get("required", []), *node.get("required", [])}
                        )
                    node.pop("required", None)
                process_node(items)

            if node.get("properties") and isinstance(node.get("properties"), dict):
                for k, v in list(node["properties"].items()):
                    node["properties"][k] = process_node(v)

            return node

        return process_node(new_schema)
