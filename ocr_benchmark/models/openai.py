import requests
from pathlib import Path
import time
import asyncio
import traceback
import json
from typing import Any, Optional, Dict
import base64

from .base import BaseModel
from .shared import OCR_SYSTEM_PROMPT, JSON_EXTRACTION_SYSTEM_PROMPT, IMAGE_EXTRACTION_SYSTEM_PROMPT
from .shared.token_cost import calculate_token_cost
from ..utils import get_mime_type
from ..types.model import ExtractionResult

from openai import OpenAI


# Helper to extract token counts from typical OpenAI response shapes
def _extract_openai_tokens(response: Any) -> tuple[int, int]:
    input_tokens = None
    output_tokens = None

    try:
        u = response.usage
        input_tokens = u.input_tokens
        output_tokens = u.output_tokens
    except Exception:
        pass

    try:
        in_t = int(input_tokens) if input_tokens is not None else 0
    except Exception:
        in_t = 0
    try:
        out_t = int(output_tokens) if output_tokens is not None else 0
    except Exception:
        out_t = 0

    return in_t, out_t


# Helper: robustly extract JSON from a text blob that may include markdown or extra prose
def _extract_json_from_text(text: str) -> Any:
    import re

    if text is None:
        raise ValueError("No text to extract JSON from")

    s = text.strip()

    # 1) Prefer ```json ... ``` or ``` ... ``` fenced blocks
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
    if m:
        candidate = m.group(1).strip()
    else:
        # 2) Otherwise look for the first {...} block
        m = re.search(r"(\{[\s\S]*\})", s)
        candidate = m.group(1).strip() if m else s

    # Try multiple parse strategies
    def try_load(js: str):
        try:
            return json.loads(js)
        except Exception:
            return None

    # direct attempt
    out = try_load(candidate)
    if out is not None:
        return out

    # strip code fences if still present
    cand = re.sub(r"^```.*?\n", "", candidate)
    cand = re.sub(r"\n```$", "", cand)
    cand = cand.strip()
    out = try_load(cand)
    if out is not None:
        return out

    # try common fixes: replace single quotes with double quotes and remove trailing commas
    cand2 = cand.replace("'", '"')
    cand2 = re.sub(r",\s*([}\]])", r"\1", cand2)
    out = try_load(cand2)
    if out is not None:
        return out

    # fallback: extract between first { and last }
    si = s.find("{")
    ei = s.rfind("}")
    if si != -1 and ei != -1 and ei > si:
        cand3 = s[si : ei + 1]
        out = try_load(cand3)
        if out is not None:
            return out

    raise ValueError("Could not extract valid JSON from text")


class OpenAIProvider(BaseModel):
    """Provider for OpenAI-compatible endpoints (e.g., compatible OpenAI servers).

    Uses `COMPATIBLE_OPENAI_API_KEY` and optional `COMPATIBLE_OPENAI_BASE_URL`.
    The implementation is defensive and tries multiple client shapes so it can
    run even if the installed OpenAI client has a different API surface.
    """

    def __init__(self, model: str, output_dir: Optional[str] = None):
        super().__init__(model, output_dir)
        try:
            # Newer OpenAI client accepts api_key and base_url
            self.client = OpenAI()
        except Exception:
            self.client = None
            raise RuntimeError("OpenAI client initialization failed")

    async def ocr(self, image_path: str) -> ExtractionResult:
        t0 = time.perf_counter()

        # Build the message payload similar to the TS implementation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_path}},
                ],
            }
        ]

        def _sync_call():
            # Try a few common client shapes
            try:
                if self.client is not None:
                    # prefer: client.chat.completions.create
                    chat = getattr(self.client, "chat", None)
                    if (
                        chat is not None
                        and hasattr(chat, "completions")
                        and hasattr(chat.completions, "create")
                    ):
                        return chat.completions.create(
                            model=self.model, messages=messages
                        )

                    # else: client.chat.create
                    if chat is not None and hasattr(chat, "create"):
                        return chat.create(model=self.model, messages=messages)

                    # else: client.chat.completions.create at top level
                    if hasattr(self.client, "chat_completions") and hasattr(
                        self.client.chat_completions, "create"
                    ):
                        return self.client.chat_completions.create(
                            model=self.model, messages=messages
                        )

                    # fallback to any 'create' on client passing parameters
                    if hasattr(self.client, "create"):
                        return self.client.create(model=self.model, messages=messages)

                # Fallback: use openai module ChatCompletion if available
                import openai as _openai

                if hasattr(_openai, "ChatCompletion") and hasattr(
                    _openai.ChatCompletion, "create"
                ):
                    return _openai.ChatCompletion.create(
                        model=self.model, messages=messages
                    )

                # As a last resort try _openai.chat.create (some newer libs)
                if hasattr(_openai, "chat") and hasattr(_openai.chat, "create"):
                    return _openai.chat.create(model=self.model, messages=messages)

                raise RuntimeError(
                    "No compatible OpenAI client available. Install 'openai' or supply a compatible client."
                )
            except Exception:
                raise

        try:
            response = await asyncio.to_thread(_sync_call)

            # Extract text from multiple response shapes
            text = ""
            try:
                if hasattr(response, "choices"):
                    # object-style
                    ch = response.choices
                    if isinstance(ch, (list, tuple)) and len(ch) > 0:
                        first = ch[0]
                        # Access nested message.content if available
                        msg = (
                            getattr(first, "message", None) or first.get("message")
                            if isinstance(first, dict)
                            else None
                        )
                        if msg is not None:
                            # message may have 'content' or another structure
                            if isinstance(msg, dict):
                                text = msg.get("content") or ""
                            else:
                                text = getattr(msg, "content", str(msg))
                        else:
                            # legacy ChatCompletion shape
                            text = (
                                getattr(first, "text", None) or first.get("text")
                                if isinstance(first, dict)
                                else str(first)
                            )
                elif isinstance(response, dict) and response.get("choices"):
                    ch = response["choices"]
                    if isinstance(ch, (list, tuple)) and len(ch) > 0:
                        first = ch[0]
                        msg = first.get("message") if isinstance(first, dict) else None
                        if msg:
                            text = msg.get("content") or ""
                        else:
                            text = first.get("text") or ""
                else:
                    text = str(response)
            except Exception:
                text = str(response)

            # token counts
            in_t, out_t = _extract_openai_tokens(response)
            try:
                input_cost = calculate_token_cost(self.model, "input", in_t)
                output_cost = calculate_token_cost(self.model, "output", out_t)
                total_cost = (input_cost or 0) + (output_cost or 0)
            except Exception:
                input_cost = output_cost = total_cost = None

            t1 = time.perf_counter()

            usage = {
                "duration": (t1 - t0),
                "inputTokens": in_t,
                "outputTokens": out_t,
                "totalTokens": in_t + out_t,
                "inputCost": input_cost,
                "outputCost": output_cost,
                "totalCost": total_cost,
            }

            return {"text": text or "", "imageBase64s": None, "usage": usage}
        except Exception as e:
            print("OpenAIProvider OCR error:", e)
            print(traceback.format_exc())
            raise

    async def extract_from_text(
        self, text: str, schema: Dict[str, Any]
    ) -> ExtractionResult:
        t0 = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": JSON_EXTRACTION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ]
        filtered_schema = self.convert_schema_for_openai(schema)

        def _sync_call():
            # Try a few common client shapes. Prefer responses.create to avoid passing
            # a text_format dict (the SDK expects a type/class for auto-parsing).
            try:
                if self.client is not None:
                    responses = getattr(self.client, "responses", None)
                    if responses is not None:
                        # Prefer the create endpoint which returns raw output_text we can parse
                        if hasattr(responses, "create"):
                            return responses.create(model=self.model, input=messages)
                        # Fall back to parse without text_format to avoid SDK auto-parsing
                        if hasattr(responses, "parse"):
                            return responses.parse(model=self.model, input=messages)
                raise RuntimeError("No compatible OpenAI responses API available")
            except Exception:
                raise RuntimeError("OpenAI _sync_call failed")

        try:
            response = await asyncio.to_thread(_sync_call)

            # Extract json from response
            json_obj = json.loads(response.output_text)

            # token counts
            in_t, out_t = _extract_openai_tokens(response)
            try:
                input_cost = calculate_token_cost(self.model, "input", in_t)
                output_cost = calculate_token_cost(self.model, "output", out_t)
                total_cost = (input_cost or 0) + (output_cost or 0)
            except Exception:
                input_cost = output_cost = total_cost = None

            t1 = time.perf_counter()

            usage = {
                "duration": (t1 - t0),
                "inputTokens": in_t,
                "outputTokens": out_t,
                "totalTokens": in_t + out_t,
                "inputCost": input_cost,
                "outputCost": output_cost,
                "totalCost": total_cost,
            }

            return {"json": json_obj, "usage": usage}
        except Exception as e:
            print("OpenAIProvider extract_from_text error:", e)
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
            b64 = base64.b64encode(data).decode("utf-8")

            image_part = None
            try:
                image_part = {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:{get_mime_type(image_path)};base64,{b64}",
                        }
                    ]
                }
            except Exception:
                image_part = {
                    "inlineData": {"type_data": type(data).__name__, "mimeType": get_mime_type(image_path)}
                }

            messages = [
                {
                    "role": "system",
                    "content": IMAGE_EXTRACTION_SYSTEM_PROMPT
                },
                image_part
            ]

            filtered_schema = self.convert_schema_for_openai(schema)

            response = await asyncio.to_thread(
                lambda: (
                    self.client.responses.create(
                        model=self.model, 
                        input=messages,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "output_format",
                                "schema": filtered_schema,
                                "strict": True,
                            }
                        },
                    )
                    if self.client is not None else None
                )
            )

            jtext = response.output_text

            try:
                json_obj = json.loads(jtext)
            except Exception as e:
                print(type(e).__name__, ": ", str(e))
                try:
                    json_obj = _extract_json_from_text(jtext)
                except Exception:
                    print("OpenAIProvider extract_from_image JSON parse failed. Raw response:\n", jtext)
                    raise
            end = time.perf_counter()

            usage = dict(response.usage)

            duration = end - t0

            in_t, out_t = _extract_openai_tokens(response)
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

            return {"json": json_obj, "usage": merged_usage}
        except Exception as e:
            print("OpenAI extract_from_image error:", e)
            print(traceback.format_exc())
            raise

    def convert_schema_for_openai(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        new_schema = json.loads(json.dumps(schema))

        def process_node(node: Any):
            if not node or not isinstance(node, dict):
                return node

            if node.get("type") == "enum" and node.get("enum") is not None:
                node["type"] = "string"
            if "enum" in node and not node.get("type"):
                node["type"] = "string"

            # Ensure object-like nodes explicitly forbid additional properties
            # and include a `required` array listing every property key (OpenAI expects this)
            if node.get("type") == "object" or node.get("properties"):
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict):
                    # OpenAI's schema validator expects `required` to be present and include
                    # every key defined under `properties` in object schemas.
                    node["required"] = list(props.keys())

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
