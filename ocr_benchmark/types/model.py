from typing import Any, Dict, TypedDict, Optional


class Usage(TypedDict, total=False):
    duration: float
    inputTokens: int
    outputTokens: int
    totalTokens: int
    inputCost: Optional[float]
    outputCost: Optional[float]
    totalCost: Optional[float]
    # provider-specific sections
    ocr: Optional[Dict[str, Any]]
    extraction: Optional[Dict[str, Any]]


class ExtractionResult(TypedDict, total=False):
    json: Optional[Dict[str, Any]]
    text: Optional[str]
    usage: Usage


__all__ = ["Usage", "ExtractionResult"]
