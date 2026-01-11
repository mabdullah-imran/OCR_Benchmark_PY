import pytest
from ocr_benchmark.evaluation.json import (
    calculate_json_accuracy,
    count_total_fields,
)


def test_identical_json():
    a = {"a": 1, "b": {"x": "foo"}}
    res = calculate_json_accuracy(a, a)
    assert res["score"] == 1.0
    assert res["jsonDiff"] == {}
    assert res["jsonDiffStats"]["total"] == 0


def test_addition():
    actual = {"a": 1}
    predicted = {"a": 1, "b": 2}
    res = calculate_json_accuracy(actual, predicted)
    assert res["jsonDiff"]["b__added"] == 2
    assert res["jsonDiffStats"]["additions"] == 1
    assert res["score"] == 0.0  # 1 addition / 1 total field


def test_modification():
    actual = {"a": 1, "b": {"x": 1}}
    predicted = {"a": 2, "b": {"x": 1}}
    res = calculate_json_accuracy(actual, predicted)
    assert res["jsonDiff"]["a"]["__old"] == 1
    assert res["jsonDiff"]["a"]["__new"] == 2
    # total fields = 2 (a and b.x), 1 modification => score 0.5
    assert pytest.approx(res["score"], rel=1e-3) == 0.5


def test_list_diffs():
    actual = {"arr": [1, 2]}
    predicted = {"arr": [1, 3, 4]}
    res = calculate_json_accuracy(actual, predicted)
    # Expect one modification and one addition (3 replaces 2, 4 added)
    assert any(elem[0] == "~" for elem in res["jsonDiff"]["arr"]) or isinstance(
        res["jsonDiff"]["arr"], list
    )
    stats = res["jsonDiffStats"]
    assert stats["additions"] >= 1
    assert stats["modifications"] >= 1


def test_count_total_fields_examples():
    obj = {"a": 1, "b": {"c": 2, "d": [3, {"e": 4}]}}
    assert count_total_fields(obj) == 4

    obj = {"a": [1, 2, 3], "b": "test", "c": True}
    assert count_total_fields(obj) == 5

    obj = {"a": [{"b": 1}, {"c": 2}], "d": "test", "e": True}
    assert count_total_fields(obj) == 4

    obj = {"a": None, "b": {"c": None}, "d": "test"}
    assert count_total_fields(obj) == 3

    obj = {"a": 1, "b__deleted": True, "c__added": "test", "d": {"e": 2}}
    assert count_total_fields(obj) == 2


def test_calculate_json_accuracy_examples():
    res = calculate_json_accuracy({"a": 1, "b": 2}, {"a": 1, "b": 3})
    assert res["score"] == 0.5

    actual = {"a": 1, "b": {"c": 2, "d": 4, "e": 4}}
    predicted = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 0.75

    actual = {"a": 1, "b": [{"c": 2, "d": 4, "e": 4, "f": [2, 9]}]}
    predicted = {"a": 1, "b": [{"c": 2, "d": 4, "e": 5, "f": [2, 3]}]}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 0.5

    # array order should not matter for arrays of objects
    actual = {
        "a": 1,
        "b": [
            {"c": 1, "d": 2},
            {"c": 3, "d": 4},
        ],
    }
    predicted = {
        "a": 1,
        "b": [
            {"c": 3, "d": 4},
            {"c": 1, "d": 2},
        ],
    }
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 1

    actual = {"a": 1, "b": [1, 2, 3]}
    predicted = {"a": 1, "b": None}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 1 / 4

    actual = {"a": 1, "b": [{"c": 1, "d": 1}, {"c": 2}, {"c": 3, "e": 4}]}
    predicted = {"a": 1, "b": None}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == round(1 / 6, 4)

    actual = {"a": 1, "b": {"c": 1, "d": {"e": 1, "f": 2}}}
    predicted = {"a": 1, "b": {"c": 1, "d": None}}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 0.5

    # null comparisons
    actual = {"a": [{"b": 1, "c": None}]}
    predicted = {"a": [{"b": 1, "c": 2}]}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 0.5

    actual = {"a": [{"b": 1, "c": None, "f": 4}]}
    predicted = {"a": [{"b": 1, "c": {"d": 2}, "f": 4}]}
    res = calculate_json_accuracy(actual, predicted)
    assert round(res["score"], 4) == 0.6667

    actual = {"a": [{"b": 1, "c": None, "f": 4}]}
    predicted = {"a": [{"b": 1, "c": {"d": 2, "e": 3}, "f": 4}]}
    res = calculate_json_accuracy(actual, predicted)
    assert round(res["score"], 4) == 0.3333

    actual = {"a": [{"b": 1, "c": None, "f": 4}]}
    predicted = {"a": [{"b": 1, "c": [3], "f": 4}]}
    res = calculate_json_accuracy(actual, predicted)
    assert round(res["score"], 4) == 0.6667

    actual = {"a": [{"b": 1, "c": 2}]}
    predicted = {"a": [{"b": 1, "c": None}]}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 0.5

    actual = {"a": [{"b": 1, "c": {"d": 2}}]}
    predicted = {"a": [{"b": 1, "c": None}]}
    res = calculate_json_accuracy(actual, predicted)
    assert res["score"] == 0.5

    actual = {"a": [{"b": 1, "c": {"d": 2, "e": 3}}]}
    predicted = {"a": [{"b": 1, "c": None}]}
    res = calculate_json_accuracy(actual, predicted)
    assert round(res["score"], 4) == 0.3333

    actual = {"a": [{"b": 1, "c": [3, 2]}]}
    predicted = {"a": [{"b": 1, "c": None}]}
    res = calculate_json_accuracy(actual, predicted)
    assert round(res["score"], 4) == 0.3333
