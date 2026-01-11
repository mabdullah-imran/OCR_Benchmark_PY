from typing import Any, Dict, Type
from pydantic import BaseModel, create_model, Field


def _map_simple_type(t: str):
    mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }
    return mapping.get(t, Any)


def generate_pydantic_model(
    schema_def: Dict[str, Any], model_name: str = "GeneratedModel"
) -> Type[BaseModel]:
    """Generate a pydantic model class from a simplified JSON schema.

    This is not a complete JSON Schema implementation but mirrors the
    behavior of the TS `generateZodSchema` helper: it supports `string`,
    `number`, `integer`, `boolean`, `object`, `array`, and `enum`.

    All properties are made Optional (nullable) by default to match the
    Zod `.nullable()` behavior in the original implementation.
    """

    properties = schema_def.get("properties", {}) or {}
    fields: Dict[str, tuple] = {}

    for key, value in properties.items():
        # Determine base type
        typ = Any

        # enum -> treat as string with description containing enum (Literal could be used but is harder to build dynamically)
        if (
            value.get("enum")
            and isinstance(value.get("enum"), list)
            and len(value.get("enum")) > 0
        ):
            typ = str
        else:
            t = value.get("type")
            if t == "array":
                items = value.get("items", {})
                item_type = Any
                if items.get("type") == "object":
                    nested = generate_pydantic_model(
                        items, model_name=f"{model_name}_{key}_Item"
                    )
                    item_type = nested
                else:
                    item_type = _map_simple_type(items.get("type"))
                from typing import List as _List

                typ = _List[item_type]  # type: ignore
            elif t == "object":
                nested = generate_pydantic_model(
                    value, model_name=f"{model_name}_{key}"
                )
                typ = nested
            else:
                typ = _map_simple_type(t)

        # Make nullable by default (Optional)
        from typing import Optional as _Optional

        field_type = _Optional[typ]  # type: ignore

        # Build Field with description if present
        description = value.get("description")
        if description:
            fields[key] = (field_type, Field(None, description=description))
        else:
            fields[key] = (field_type, None)

    # Create pydantic model with strict behavior (forbid extra fields)
    class Config:
        extra = "forbid"

    Model = create_model(model_name, __config__=Config, **fields)
    return Model


__all__ = ["generate_pydantic_model"]
