from typing import Any, Dict


def calculate_json_accuracy(
    actual: Dict[str, Any], predicted: Dict[str, Any], ignore_cases: bool = False
) -> Dict[str, Any]:
    """Calculate accuracy between two JSON-like dicts.

    Returns a dict matching the TS AccuracyResult shape:
      {
        "score": float,
        "jsonDiff": Dict[str, Any],
        "fullJsonDiff": Dict[str, Any],
        "jsonDiffStats": { additions, deletions, modifications, total },
        "totalFields": int,
      }
    """
    processed_actual = convert_strings_to_uppercase(actual) if ignore_cases else actual
    processed_predicted = (
        convert_strings_to_uppercase(predicted) if ignore_cases else predicted
    )

    full_diff = _diff_json(processed_actual, processed_predicted, full=True)
    diff_result = _diff_json(processed_actual, processed_predicted, full=False)
    total_fields = count_total_fields(processed_actual)

    if not diff_result:
        return {
            "score": 1.0,
            "jsonDiff": {},
            "fullJsonDiff": {},
            "jsonDiffStats": {
                "additions": 0,
                "deletions": 0,
                "modifications": 0,
                "total": 0,
            },
            "totalFields": total_fields,
        }

    changes = count_changes(diff_result)

    if total_fields == 0:
        score = 0.0 if (changes["total"] > 0) else 1.0
    else:
        score = max(
            0.0,
            1
            - (changes["additions"] + changes["deletions"] + changes["modifications"])
            / total_fields,
        )

    return {
        "score": round(score, 4),
        "jsonDiff": diff_result,
        "fullJsonDiff": full_diff,
        "jsonDiffStats": changes,
        "totalFields": total_fields,
    }


def convert_strings_to_uppercase(obj: Any) -> Any:
    if obj is None or not isinstance(obj, (dict, list)):
        if isinstance(obj, str):
            return obj.upper()
        return obj

    if isinstance(obj, list):
        return [convert_strings_to_uppercase(i) for i in obj]

    result: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, str):
            result[k] = v.upper()
        elif isinstance(v, (dict, list)):
            result[k] = convert_strings_to_uppercase(v)
        else:
            result[k] = v
    return result


def _diff_json(a: Any, b: Any, full: bool = False) -> Dict[str, Any]:
    """Return a JSON-diff like structure.

    rich "full" format:
      - dict keys present only in a => { key: {"__op":"-","__old": value} }
      - dict keys present only in b => { key: {"__op":"+","__new": value} }
      - modifications => { key: {"__op":"~","__old": old, "__new": new} }
      - nested diffs are placed under "diff" for clearer presentation when full=True

    non-full (compact) format preserves the existing shape used by our change counting:
      - deletions: key__deleted: value
      - additions: key__added: value
      - modifications: key: {"__old": old, "__new": new} or array [op, element]
    """
    if a is None and b is None:
        return {}

    # Primitives
    if not isinstance(a, (dict, list)) and not isinstance(b, (dict, list)):
        if a == b:
            return {}
        if full:
            return {"__op": "~", "__old": a, "__new": b}
        return {"__old": a, "__new": b}

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        result: Dict[str, Any] = {}
        keys = set(a.keys()) | set(b.keys())
        for key in sorted(keys):
            in_a = key in a
            in_b = key in b
            if in_a and not in_b:
                if full:
                    result[key] = {"__op": "-", "__old": a[key]}
                else:
                    result[f"{key}__deleted"] = a[key]
            elif in_b and not in_a:
                if full:
                    result[key] = {"__op": "+", "__new": b[key]}
                else:
                    result[f"{key}__added"] = b[key]
            else:
                va = a[key]
                vb = b[key]
                if isinstance(va, dict) and isinstance(vb, dict):
                    child = _diff_json(va, vb, full=full)
                    if child:
                        if full:
                            result[key] = {"__op": "~", "diff": child}
                        else:
                            result[key] = child
                elif isinstance(va, list) and isinstance(vb, list):
                    child = _diff_json(va, vb, full=full)
                    if child:
                        if full:
                            result[key] = {"__op": "~", "diff": child}
                        else:
                            result[key] = child
                elif va == vb:
                    continue
                else:
                    if full:
                        result[key] = {"__op": "~", "__old": va, "__new": vb}
                    else:
                        result[key] = {"__old": va, "__new": vb}
        return result

    # Lists
    if isinstance(a, list) and isinstance(b, list):
        # If lists contain dicts we can attempt unordered matching (multisets)
        if (
            len(a) == len(b)
            and all(isinstance(x, dict) for x in a)
            and all(isinstance(x, dict) for x in b)
        ):
            b_copy = list(b)
            matched = True
            for va in a:
                found_idx = None
                for idx, vb in enumerate(b_copy):
                    # consider them equal if diff is empty in compact mode
                    if _diff_json(va, vb, full=False) == {}:
                        found_idx = idx
                        break
                if found_idx is None:
                    matched = False
                    break
                b_copy.pop(found_idx)
            if matched and not b_copy:
                return []

        # For lists that are not lists-of-dicts:
        # - if they are identical, no diff
        # - if different lengths, fall back to index-wise comparison to detect additions/deletions
        # - if same length but unequal, treat as modification of the whole list (so changes are counted by list size)
        if not (
            all(isinstance(x, dict) for x in a) and all(isinstance(x, dict) for x in b)
        ):
            if a == b:
                return []
            if len(a) != len(b):
                # fall through to index-wise comparison to detect additions/deletions
                pass
            else:
                if full:
                    return {"__op": "~", "__old": a, "__new": b}
                return {"__old": a, "__new": b}

        # Otherwise (lists of dicts or mismatched-length primitive lists) fall back to index-wise comparison
        result: list = []
        la = len(a)
        lb = len(b)
        maxl = max(la, lb)
        for i in range(maxl):
            in_a = i < la
            in_b = i < lb
            if in_a and not in_b:
                if full:
                    result.append({"op": "-", "value": a[i]})
                else:
                    result.append(["-", a[i]])
            elif in_b and not in_a:
                if full:
                    result.append({"op": "+", "value": b[i]})
                else:
                    result.append(["+", b[i]])
            else:
                va = a[i]
                vb = b[i]
                if isinstance(va, dict) and isinstance(vb, dict):
                    child = _diff_json(va, vb, full=full)
                    if child:
                        if full:
                            result.append({"op": "~", "diff": child})
                        else:
                            result.append(["~", child])
                elif isinstance(va, list) and isinstance(vb, list):
                    child = _diff_json(va, vb, full=full)
                    if child:
                        if full:
                            result.append({"op": "~", "diff": child})
                        else:
                            result.append(["~", child])
                elif va == vb:
                    continue
                else:
                    if full:
                        result.append({"op": "~", "value": {"__old": va, "__new": vb}})
                    else:
                        result.append(["~", {"__old": va, "__new": vb}])
        return result

    # Different types -> represent as modification
    if full:
        return {"__op": "~", "__old": a, "__new": b}
    return {"__old": a, "__new": b}


def count_changes(diff_result: Any) -> Dict[str, int]:
    changes = {"additions": 0, "deletions": 0, "modifications": 0, "total": 0}

    def traverse(obj: Any) -> None:
        if not obj or not isinstance(obj, (dict, list)):
            return

        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                op, element = item[0], item[1]
                if not isinstance(element, dict):
                    if op == "+":
                        changes["additions"] += 1
                    elif op == "-":
                        changes["deletions"] += 1
                    elif op == "~":
                        changes["modifications"] += 1
                else:
                    if op == "+":
                        changes["additions"] += count_total_fields(element)
                    elif op == "-":
                        changes["deletions"] += count_total_fields(element)
                    elif op == "~":
                        traverse(element)
            return

        # dict
        # If the dict itself is a modification box like {"__old": ..., "__new": ...}
        if "__old" in obj and "__new" in obj:
            old = obj.get("__old")
            new = obj.get("__new")
            if old is None and new is not None:
                changes["modifications"] += count_total_fields(new) or 1
            else:
                changes["modifications"] += count_total_fields(old) or 1
            return

        for key, value in obj.items():
            if key.endswith("__deleted"):
                if value is None or not isinstance(value, (dict, list)):
                    changes["deletions"] += 1
                else:
                    changes["deletions"] += count_total_fields(value)
            elif key.endswith("__added"):
                if value is None or not isinstance(value, (dict, list)):
                    changes["additions"] += 1
                else:
                    changes["additions"] += count_total_fields(value)
            elif isinstance(value, dict) and value is not None:
                if "__old" in value and "__new" in value:
                    old = value.get("__old")
                    new = value.get("__new")
                    if old is None and new is not None:
                        changes["modifications"] += count_total_fields(new) or 1
                    else:
                        changes["modifications"] += count_total_fields(old) or 1
                else:
                    traverse(value)
            elif isinstance(value, list):
                traverse(value)

    traverse(diff_result)
    changes["total"] = (
        changes["additions"] + changes["deletions"] + changes["modifications"]
    )
    return changes


def count_total_fields(obj: Any) -> int:
    count = 0

    def traverse(current: Any) -> None:
        nonlocal count
        if current is None or not isinstance(current, (dict, list)):
            return

        if isinstance(current, list):
            for item in current:
                if isinstance(item, dict):
                    traverse(item)
                else:
                    count += 1
            return

        for key, value in current.items():
            if "__" in key:
                continue
            if value is None or isinstance(value, (str, bool, int, float)):
                count += 1
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        traverse(item)
                    else:
                        count += 1
            elif isinstance(value, dict):
                traverse(value)

    traverse(obj)
    return count


__all__ = [
    "calculate_json_accuracy",
    "convert_strings_to_uppercase",
    "count_changes",
    "count_total_fields",
]
