from typing import Dict, List

# Token cost mapping ported from tokenCost.ts
TOKEN_COST: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash-lite": {"input":0.1, "output":0.4},
    "gemini-2.5-flash-lite-preview-09-2025": {"input":0.1, "output":0.4},
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-2.5-flash-preview-09-2025": {"input": 0.3, "output": 2.5},
    "gemini-2.5-pro": {"input": 1.25, "output": 10},
    "gemini-3-flash-preview": {"input": 0.5, "output": 3},
    "gemini-3-pro-preview": {"input": 2, "output": 12},
    "gpt-4o": {"input": 2.5, "output": 10},
    "gpt-4o-2024-11-20": {"input": 2.5, "output": 10},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2, "output": 8},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-5-nano": {"input": 0.05, "output": 0.4},
    "gpt-5-mini": {"input": 0.25, "output": 2},
    "gpt-5": {"input": 1.25, "output": 10},
    "gpt-5.2": {"input": 1.75, "output": 14},
    "gpt-5.2-pro": {"input": 21, "output": 168},
    "o1": {"input": 15, "output": 60},
    "o1-mini": {"input": 1.1, "output": 4.4},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "o3-pro": {"input": 20, "output": 80},
    "o4-mini": {"input": 1.1, "output": 4.4},
    "chatgpt-4o-latest": {"input": 2.5, "output": 10},
}

# Support FINETUNED_MODELS if present in a registry module
try:
    from ..registry import FINETUNED_MODELS  # type: ignore
except Exception:
    FINETUNED_MODELS: List[str] = []


def calculate_token_cost(model: str, type_: str, tokens: int) -> float:
    """Calculate cost in dollars for given model, token type and token count.

    - `type_` must be 'input' or 'output'.
    - Returns cost scaled by 1e6 as in the original implementation.
    """
    if type_ not in ("input", "output"):
        raise ValueError("type_ must be 'input' or 'output'")

    fine_tune_cost = {m: {"input": 3.75, "output": 15.0} for m in FINETUNED_MODELS}
    combined_cost: Dict[str, Dict[str, float]] = {**TOKEN_COST, **fine_tune_cost}

    model_info = combined_cost.get(model)
    if not model_info:
        raise KeyError(f"Model '{model}' is not supported.")

    return (model_info[type_] * tokens) / 1_000_000


__all__ = ["TOKEN_COST", "calculate_token_cost"]
