import importlib.util
import os

# Import the evaluation module directly by path to avoid importing package-level dependencies
here = os.path.dirname(os.path.dirname(__file__))
module_path = os.path.join(here, "ocr_benchmark", "evaluation", "json.py")

spec = importlib.util.spec_from_file_location("eval_json", module_path)
mod = importlib.util.module_from_spec(spec)
assert isinstance(spec.loader, importlib.util.LazyLoader) or True
spec.loader.exec_module(mod)  # type: ignore

calculate_json_accuracy = mod.calculate_json_accuracy


def assert_eq(a, b, msg=None):
    if a != b:
        raise AssertionError(msg or f"{a} != {b}")


def test_identical_json():
    a = {"a": 1, "b": {"x": "foo"}}
    res = calculate_json_accuracy(a, a)
    assert_eq(res["score"], 1.0)
    assert_eq(res["jsonDiff"], {})
    assert_eq(res["jsonDiffStats"]["total"], 0)


def test_addition():
    actual = {"a": 1}
    predicted = {"a": 1, "b": 2}
    res = calculate_json_accuracy(actual, predicted)
    assert res["jsonDiff"]["b__added"] == 2
    assert res["jsonDiffStats"]["additions"] == 1
    assert res["score"] == 0.0


def test_modification():
    actual = {"a": 1, "b": {"x": 1}}
    predicted = {"a": 2, "b": {"x": 1}}
    res = calculate_json_accuracy(actual, predicted)
    assert res["jsonDiff"]["a"]["__old"] == 1
    assert res["jsonDiff"]["a"]["__new"] == 2
    assert abs(res["score"] - 0.5) < 1e-3


def test_list_diffs():
    actual = {"arr": [1, 2]}
    predicted = {"arr": [1, 3, 4]}
    res = calculate_json_accuracy(actual, predicted)
    assert "arr" in res["jsonDiff"]
    stats = res["jsonDiffStats"]
    assert stats["additions"] >= 1
    assert stats["modifications"] >= 1


if __name__ == "__main__":
    tests = [
        test_identical_json,
        test_addition,
        test_modification,
        test_list_diffs,
    ]

    for t in tests:
        try:
            t()
            print(f"{t.__name__}: PASS")
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
            raise
    print("All tests finished.")
