import importlib.util
import os

here = os.path.dirname(os.path.dirname(__file__))
module_path = os.path.join(here, "ocr_benchmark", "evaluation", "json.py")
spec = importlib.util.spec_from_file_location("eval_json", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

calculate_json_accuracy = mod.calculate_json_accuracy

actual = {"arr": [1, 2]}
pred = {"arr": [1, 3, 4]}
res = calculate_json_accuracy(actual, pred)
import json

print("diff:", json.dumps(res["jsonDiff"], indent=2))
print("stats:", res["jsonDiffStats"])
print("full:", json.dumps(res["fullJsonDiff"], indent=2))
print("score:", res["score"])
