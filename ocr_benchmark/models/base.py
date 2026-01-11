"""Base provider classes and typing protocols for model adapters.

This module provides an async-friendly abstract base class `BaseModel`
that mirrors the TypeScript `ModelProvider` interface, plus a
`ModelProviderProtocol` for structural typing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from ..types.model import ExtractionResult


class BaseModel(ABC):
    """Base Adapter Class for OCR and extraction providers.

    Implementations should provide at least `ocr`. `extract_from_text`
    and `extract_from_image` are optional but commonly implemented by
    extraction-capable providers.
    """

    def __init__(self, model: str, output_dir: Optional[str] = None):
        self.model = model
        self.output_dir = output_dir

    @abstractmethod
    async def ocr(self, image_path: str) -> ExtractionResult:
        """Perform OCR on the given image.

        Returns an ExtractionResult with keys:
            - text: Optional[str]
            - imageBase64s: Optional[List[str]]
            - usage: Usage
        """
        raise NotImplementedError

    async def extract_from_text(
        self,
        text: str,
        schema: Dict[str, Any],
        imageBase64s: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Optional: produce JSON from text + schema.

        Default behaviour is to raise NotImplementedError so providers
        that do not support text->JSON extraction can keep the method
        absent.
        """
        raise NotImplementedError(
            "extract_from_text not implemented for this provider."
        )

    async def extract_from_image(
        self, image_path: str, schema: Dict[str, Any]
    ) -> ExtractionResult:
        """Optional: produce JSON directly from an image.

        Providers that support direct image->JSON extraction should
        override this method.
        """
        raise NotImplementedError(
            "extract_from_image not implemented for this provider."
        )


@runtime_checkable
class ModelProviderProtocol(Protocol):
    """Structural typing for model providers.

    Use this protocol when you want to accept any object that implements
    the expected async methods and attributes, without requiring
    inheritance from `BaseModel`.
    """

    model: str
    output_dir: Optional[str]

    async def ocr(self, image_path: str) -> ExtractionResult: ...

    async def extract_from_text(
        self,
        text: str,
        schema: Dict[str, Any],
        imageBase64s: Optional[List[str]] = None,
    ) -> ExtractionResult: ...

    async def extract_from_image(
        self, image_path: str, schema: Dict[str, Any]
    ) -> ExtractionResult: ...
