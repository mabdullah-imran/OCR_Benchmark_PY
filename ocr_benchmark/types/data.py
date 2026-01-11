from typing import Any, Dict, TypedDict, Optional
from .model import Usage


class Metadata(TypedDict, total=False):
    orientation: Optional[int]
    documentQuality: Optional[str]
    resolution: Optional[list]
    language: Optional[str]


class JsonSchema(TypedDict, total=False):
    type: str
    description: Optional[str]
    properties: Optional[Dict[str, "JsonSchema"]]
    items: Optional["JsonSchema"]
    required: Optional[list]


class Input(TypedDict):
    imageUrl: str
    metadata: Metadata
    jsonSchema: JsonSchema
    trueJsonOutput: Dict[str, Any]
    trueMarkdownOutput: str


class AccuracyResult(TypedDict, total=False):
    score: Optional[float]
    jsonDiff: Optional[Dict[str, Any]]
    fullJsonDiff: Optional[Dict[str, Any]]
    jsonDiffStats: Optional[Dict[str, Any]]


class Result(TypedDict, total=False):
    fileUrl: str
    metadata: Metadata
    jsonSchema: JsonSchema
    ocrModel: str
    extractionModel: str
    directImageExtraction: Optional[bool]
    trueMarkdown: Optional[str]
    trueJson: Optional[Dict[str, Any]]
    predictedMarkdown: Optional[str]
    predictedJson: Optional[Dict[str, Any]]
    jsonAccuracy: Optional[float]
    jsonDiff: Optional[Dict[str, Any]]
    fullJsonDiff: Optional[Dict[str, Any]]
    jsonDiffStats: Optional[Dict[str, Any]]
    jsonAccuracyResult: Optional[AccuracyResult]
    usage: Optional[Usage]
    error: Optional[Any]


__all__ = ["Input", "Metadata", "JsonSchema", "Result", "AccuracyResult"]
