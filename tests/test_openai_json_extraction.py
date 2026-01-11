import pytest
from ocr_benchmark.models.openai import _extract_json_from_text


def test_extract_from_fenced_json():
    txt = "Here you go:\n```json\n{\"a\": 1, \"b\": 2}\n```"
    assert _extract_json_from_text(txt) == {"a": 1, "b": 2}


def test_extract_from_plain_embedded_json():
    txt = "Response: {\"x\": 3, \"y\": [1,2,3]} End."
    assert _extract_json_from_text(txt) == {"x": 3, "y": [1, 2, 3]}


def test_extract_with_single_quotes_and_trailing_comma():
    txt = "```\n{'a': 1, 'b': 2,}\n```"
    assert _extract_json_from_text(txt) == {"a": 1, "b": 2}


def test_missing_json_raises():
    with pytest.raises(Exception):
        _extract_json_from_text("No JSON here")
