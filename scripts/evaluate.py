"""Score structured-output predictions against local annotated cases."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ARRAY_FIELDS = ("/addresses", "/practice_areas", "/coverage_terms")
MISSING = object()


def flatten(value: object, path: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        out = {}
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            out.update(flatten(child, f"{path}/{escaped}"))
        return out
    if isinstance(value, list) and value:
        out = {}
        for index, child in enumerate(value):
            out.update(flatten(child, f"{path}/{index}"))
        return out
    return {path or "/": value}


def lookup(value: object, pointer: str) -> object:
    current = value
    if pointer in ("", "/"):
        return current
    for part in pointer.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return MISSING
    return current


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if item == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def score_case(case: dict, prediction: dict, validator: Draft202012Validator) -> dict:
    expected = case["expected"]
    actual = prediction.get("parsed")
    parse_error = prediction.get("parse_error")
    schema_errors = []
    if actual is not None:
        schema_errors = [
            {"path": "/" + "/".join(map(str, error.absolute_path)), "message": error.message}
            for error in validator.iter_errors(actual)
        ]

    valid = actual is not None and not schema_errors
    target_fields = flatten(expected)
    predicted_fields = flatten(actual) if actual is not None else {}
    true_positive = false_positive = false_negative = 0
    hallucinations = absent_fields = 0
    mismatches = []

    for pointer, target in target_fields.items():
        found = lookup(actual, pointer) if actual is not None else MISSING
        if target is None:
            absent_fields += 1
            if found is not MISSING and found is not None:
                hallucinations += 1
                false_positive += 1
                mismatches.append({"path": pointer, "kind": "unsupported_value", "expected": None, "actual": found})
        elif found is MISSING:
            false_negative += 1
            mismatches.append({"path": pointer, "kind": "missing_value", "expected": target})
        elif found == target:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
            mismatches.append({"path": pointer, "kind": "value_mismatch", "expected": target, "actual": found})

    for pointer, value in predicted_fields.items():
        if pointer not in target_fields and value is not None:
            false_positive += 1
            mismatches.append({"path": pointer, "kind": "unexpected_value", "actual": value})

    arrays = []
    array_row_matches = predicted_row_count = expected_row_count = 0
    array_set_exact_count = array_order_exact_count = 0
    for pointer in ARRAY_FIELDS:
        target = lookup(expected, pointer)
        if not isinstance(target, list):
            continue
        found = lookup(actual, pointer) if actual is not None else MISSING
        predicted = found if isinstance(found, list) else []
        target_keys = [canonical(row) for row in target]
        predicted_keys = [canonical(row) for row in predicted]
        matched = sum((Counter(target_keys) & Counter(predicted_keys)).values())
        set_exact = Counter(target_keys) == Counter(predicted_keys)
        order_exact = target_keys == predicted_keys
        expected_row_count += len(target)
        predicted_row_count += len(predicted)
        array_row_matches += matched
        array_set_exact_count += int(set_exact)
        array_order_exact_count += int(order_exact)
        arrays.append({
            "path": pointer,
            "expected_rows": len(target),
            "predicted_rows": len(predicted),
            "matched_rows": matched,
            "exact_row_set": set_exact,
            "ordered_exact": order_exact,
            "row_order_accuracy": lcs_length(target_keys, predicted_keys) / len(target_keys) if target_keys else 1.0,
        })
        if not set_exact:
            mismatches.append({"path": pointer, "kind": "array_row_set_mismatch"})
        elif not order_exact:
            mismatches.append({"path": pointer, "kind": "array_order_mismatch"})

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row_precision = array_row_matches / predicted_row_count if predicted_row_count else (1.0 if not expected_row_count else 0.0)
    row_recall = array_row_matches / expected_row_count if expected_row_count else 1.0
    row_f1 = 2 * row_precision * row_recall / (row_precision + row_recall) if row_precision + row_recall else 0.0

    return {
        "id": case["id"],
        "group_id": case["group_id"],
        "parse_valid": actual is not None,
        "parse_error": parse_error,
        "schema_valid": valid,
        "schema_errors": schema_errors,
        "all_constraints_satisfied": valid and actual == expected,
        "fields": {"true_positive": true_positive, "false_positive": false_positive,
                   "false_negative": false_negative, "precision": precision, "recall": recall, "f1": f1},
        "missing_field_hallucination": {"unsupported_values": hallucinations, "absent_fields": absent_fields,
                                         "false_value_rate": hallucinations / absent_fields if absent_fields else None},
        "arrays": arrays,
        "array_aggregate": {"row_precision": row_precision, "row_recall": row_recall, "row_f1": row_f1,
                            "exact_row_set_rate": array_set_exact_count / len(arrays) if arrays else None,
                            "ordered_exact_rate": array_order_exact_count / len(arrays) if arrays else None},
        "mismatches": mismatches,
    }


def aggregate(case_scores: list[dict], case_count: int, group_count: int) -> dict:
    totals = {key: sum(score["fields"][key] for score in case_scores)
              for key in ("true_positive", "false_positive", "false_negative")}
    tp, fp, fn = totals["true_positive"], totals["false_positive"], totals["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    hallucinated = sum(score["missing_field_hallucination"]["unsupported_values"] for score in case_scores)
    absent = sum(score["missing_field_hallucination"]["absent_fields"] for score in case_scores)
    arrays = [array for score in case_scores for array in score["arrays"]]
    row_matches = sum(array["matched_rows"] for array in arrays)
    predicted_rows = sum(array["predicted_rows"] for array in arrays)
    expected_rows = sum(array["expected_rows"] for array in arrays)
    row_precision = row_matches / predicted_rows if predicted_rows else (1.0 if not expected_rows else 0.0)
    row_recall = row_matches / expected_rows if expected_rows else 1.0
    return {
        "case_count": case_count,
        "scored_case_count": len(case_scores),
        "unique_group_count": group_count,
        "held_out_test_set_available": False,
        "schema_validity_rate": sum(score["schema_valid"] for score in case_scores) / len(case_scores) if case_scores else None,
        "all_constraints_satisfied_rate": sum(score["all_constraints_satisfied"] for score in case_scores) / len(case_scores) if case_scores else None,
        "field_value": {**totals, "precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0},
        "missing_field_hallucination": {"unsupported_values": hallucinated, "absent_fields": absent,
                                         "false_value_rate": hallucinated / absent if absent else None},
        "arrays": {"scored_arrays": len(arrays), "expected_rows": expected_rows, "predicted_rows": predicted_rows,
                   "row_precision": row_precision, "row_recall": row_recall,
                   "row_f1": 2 * row_precision * row_recall / (row_precision + row_recall)
                   if row_precision + row_recall else 0.0,
                   "exact_row_set_rate": sum(array["exact_row_set"] for array in arrays) / len(arrays) if arrays else None,
                   "ordered_exact_rate": sum(array["ordered_exact"] for array in arrays) / len(arrays) if arrays else None},
        "latency_seconds": {"total": sum(score["latency_seconds"] for score in case_scores),
                            "mean_per_case": sum(score["latency_seconds"] for score in case_scores) / len(case_scores) if case_scores else None},
        "interpretation": "Exploratory baseline only. Cases from one insured are not independent; no untouched test set or unrelated-task regression set is available.",
    }
