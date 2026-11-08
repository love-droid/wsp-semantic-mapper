"""Re-score saved generations after removing one surrounding Markdown JSON fence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator, FormatChecker

from evaluate import aggregate, score_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data/local/development.seed.jsonl"
DEFAULT_SCHEMA = ROOT / "schemas/insurance_submission.v1.schema.json"
FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(.*?)\n```\s*\Z", re.DOTALL | re.IGNORECASE)


def parse_fenced_json(raw: str) -> tuple[object | None, str | None, bool]:
    match = FENCE.fullmatch(raw)
    normalized = match.group(1) if match else raw
    try:
        return json.loads(normalized), None, match is not None
    except json.JSONDecodeError as error:
        return None, f"{error.msg} at line {error.lineno}, column {error.colno}", match is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = {case["id"]: case for case in (
        json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()
    )}
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    case_scores = []
    for prediction in predictions:
        parsed, parse_error, fence_removed = parse_fenced_json(prediction["raw_output"])
        rescored_prediction = {**prediction, "parsed": parsed, "parse_error": parse_error}
        score = score_case(cases[prediction["id"]], rescored_prediction, validator)
        score["latency_seconds"] = prediction.get("latency_seconds", 0.0)
        score["markdown_json_fence_removed"] = fence_removed
        case_scores.append(score)

    result = aggregate(case_scores, len(case_scores), len({score["group_id"] for score in case_scores}))
    result.update({
        "score_variant": "diagnostic-rescore-after-removing-only-one-surrounding-markdown-json-fence",
        "raw_predictions_unchanged": True,
        "human_reviewed_annotations": all(cases[score["id"]].get("human_reviewed", False) for score in case_scores),
        "case_scores": case_scores,
    })
    output = args.output or args.predictions.parent / "metrics.fence-stripped.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "schema_validity_rate": result["schema_validity_rate"],
        "field_value": result["field_value"],
        "missing_field_hallucination": result["missing_field_hallucination"],
        "all_constraints_satisfied_rate": result["all_constraints_satisfied_rate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
