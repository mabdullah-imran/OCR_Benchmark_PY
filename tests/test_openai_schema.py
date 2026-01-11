from ocr_benchmark.models.openai import OpenAIProvider


def test_convert_schema_for_openai_adds_required_and_disallows_additional():
    schema = {
        "type": "object",
        "properties": {
            "totals": {
                "type": "object",
                "properties": {
                    "tax": {"type": "number"},
                    "subtotal": {"type": "number"},
                },
            },
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        },
    }

    converted = OpenAIProvider.convert_schema_for_openai(None, schema)

    # top-level object should disallow additionalProperties and require 'totals' and 'items'
    assert converted.get("additionalProperties") is False
    assert set(converted.get("required", [])) >= {"totals", "items"}

    totals = converted["properties"]["totals"]
    assert totals.get("additionalProperties") is False
    # totals should have required list including both 'tax' and 'subtotal'
    assert set(totals.get("required", [])) >= {"tax", "subtotal"}

    items = converted["properties"]["items"]
    # items should still process array items and add additionalProperties=False to the item object
    assert items.get("items", {}).get("additionalProperties") is False
    assert set(items.get("items", {}).get("required", [])) >= {"name"}
