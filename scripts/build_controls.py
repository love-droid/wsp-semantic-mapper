"""Write the frozen unrelated-task and array controls. No model and no held-out data."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/controls"

ARITHMETIC = (
    (17, 28), (46, 19), (8, 7), (100, 23), (6, 15),
    (34, 34), (90, 11), (3, 41), (12, 9), (55, 16),
    (2, 2), (70, 25), (14, 5), (9, 30), (21, 21),
    (4, 18), (63, 7), (11, 11), (80, 19), (13, 6),
)
PLURALS = (
    "cat", "dog", "pen", "car", "book", "lamp", "coin", "desk", "boat", "tree",
    "cup", "star", "road", "bird", "shoe", "hill", "farm", "gate", "ship", "bell",
)
COPY_TEXTS = (
    "amber-kite-17", "north-dock-4", "quiet-river-88", "plain-stone-2", "silver-key-50",
    "round-hill-9", "open-gate-31", "blue-line-6", "dry-field-12", "west-pier-70",
    "small-bell-3", "even-path-15", "cold-lake-22", "red-post-8", "tall-oak-41",
    "soft-rain-5", "new-mill-19", "old-map-27", "fair-wind-10", "last-stop-64",
)
CODE_VALUES = (
    ("3*4", "12"), ("10-3", "7"), ("2**5", "32"), ("9//2", "4"), ("8%3", "2"),
    ("len('kite')", "4"), ("int(True)", "1"), ("max(2, 9)", "9"), ("min(4, 4)", "4"), ("abs(-6)", "6"),
    ("'ab'+'c'", "abc"), ("bool(0)", "False"), ("bool(1)", "True"), ("round(2.2)", "2"), ("pow(2, 3)", "8"),
    ("list(range(3))", "[0, 1, 2]"), ("'Hi'.lower()", "hi"), ("'ok'.upper()", "OK"), ("divmod(7, 2)[0]", "3"), ("sorted([3, 1])[0]", "1"),
)
FENCE_SNIPPETS = (
    "print(1)", "print(2)", "x = 3", "y = 4", "z = 5",
    "n = 6", "p = 7", "q = 8", "r = 9", "s = 10",
    "t = 11", "u = 12", "v = 13", "w = 14", "a = 15",
    "b = 16", "c = 17", "d = 18", "e = 19", "f = 20",
)


def practice_schema():
    return {"type": "object", "additionalProperties": False, "required": ["practice_areas"],
            "properties": {"practice_areas": {"type": ["array", "null"], "items": {
                "type": "object", "additionalProperties": False, "required": ["name", "percentage"],
                "properties": {"name": {"type": "string"}, "percentage": {"type": "number"}}}}}}


def coverage_schema():
    return {"type": "object", "additionalProperties": False, "required": ["coverage_terms"],
            "properties": {"coverage_terms": {"type": ["array", "null"], "items": {
                "type": "object", "additionalProperties": False, "required": ["coverage", "term_type", "amount"],
                "properties": {"coverage": {"type": "string"}, "term_type": {"type": "string", "enum": ["limit", "deductible", "retention"]},
                               "amount": {"type": "number"}}}}}}


def address_schema():
    return {"type": "object", "additionalProperties": False, "required": ["addresses"],
            "properties": {"addresses": {"type": ["array", "null"], "items": {
                "type": "object", "additionalProperties": False, "required": ["role", "city", "postal_code"],
                "properties": {"role": {"type": "string", "enum": ["primary_office", "mailing"]},
                               "city": {"type": "string"}, "postal_code": {"type": "string"}}}}}}


def unrelated_cases():
    cases = []
    for index, (left, right) in enumerate(ARITHMETIC):
        cases.append({"id": f"arith-{index:02d}", "suite": "unrelated", "kind": "exact",
                      "prompt": f"What is {left}+{right}? Reply with digits only.", "gold": str(left + right)})
    for index, word in enumerate(PLURALS):
        cases.append({"id": f"plural-{index:02d}", "suite": "unrelated", "kind": "exact",
                      "prompt": f"Write the plural of {word}. Reply with that one word only.", "gold": word + "s"})
    for index, text in enumerate(COPY_TEXTS):
        cases.append({"id": f"copy-{index:02d}", "suite": "unrelated", "kind": "exact",
                      "prompt": f"Repeat this text exactly: {text}", "gold": text})
    for index, (expression, gold) in enumerate(CODE_VALUES):
        cases.append({"id": f"code-{index:02d}", "suite": "unrelated", "kind": "exact",
                      "prompt": f"What value does this Python expression produce: {expression}? Reply with only the value.", "gold": gold})
    for index, snippet in enumerate(FENCE_SNIPPETS):
        cases.append({"id": f"fence-{index:02d}", "suite": "unrelated", "kind": "fence",
                      "prompt": f"Show the exact Python line {snippet} inside one markdown code fence.",
                      "gold": snippet})
    return cases


def array_cases():
    areas = ("Litigation", "Estates", "Patents", "Appeals", "Mediation", "Arbitration")
    cases = []
    for index in range(30):
        first, second = areas[index % 6], areas[(index + 2) % 6]
        left, right = 20 + index, 80 - index
        rows = [{"name": first, "percentage": left}, {"name": second, "percentage": right}]
        if index % 5 == 0:
            rows.append({"name": areas[(index + 4) % 6], "percentage": 10})
            right = 70 - index
            rows[1]["percentage"] = right
        text = "Selected practice areas:\n" + "\n".join(f"- {row['name']}: {row['percentage']}%" for row in rows)
        text += f"\nPrinted but not selected: {areas[(index + 1) % 6]}"
        cases.append({"id": f"areas-{index:02d}", "group_id": f"areas-{index:02d}", "suite": "arrays",
                      "input": text, "json_schema": practice_schema(), "expected": {"practice_areas": rows}})
    for index in range(20):
        kind = ("limit", "deductible", "retention")[index % 3]
        amount = 1000 * (index + 1)
        row = {"coverage": f"Coverage {index}", "term_type": kind, "amount": amount}
        other = ("limit", "deductible", "retention")[(index + 1) % 3]
        text = (
            f"Schedule row: Coverage {index}; type {kind}; amount {amount}\n"
            f"Example row, not part of this account: Coverage SAMPLE; type {other}; amount 1"
        )
        cases.append({"id": f"terms-{index:02d}", "group_id": f"terms-{index:02d}", "suite": "arrays",
                      "input": text, "json_schema": coverage_schema(), "expected": {"coverage_terms": [row]}})
    for index in range(10):
        row = {"role": "mailing" if index % 2 else "primary_office", "city": f"Town {index}", "postal_code": f"{10000 + index}"}
        text = (
            f"Insured {row['role'].replace('_', ' ')}: {row['city']} {row['postal_code']}\n"
            f"Broker office, not the insured: Broker City {index} 99999"
        )
        cases.append({"id": f"address-{index:02d}", "group_id": f"address-{index:02d}", "suite": "arrays",
                      "input": text, "json_schema": address_schema(), "expected": {"addresses": [row]}})
    for index in range(10):
        text = f"Account {index}: the practice-area schedule was not included in this document.\nBroker note {index} is unrelated."
        cases.append({"id": f"areas-null-{index:02d}", "group_id": f"areas-null-{index:02d}", "suite": "arrays",
                      "input": text, "json_schema": practice_schema(), "expected": {"practice_areas": None}})
    return cases


def main():
    unrelated = unrelated_cases()
    arrays = array_cases()
    if len(unrelated) != 100 or len({case["id"] for case in unrelated}) != 100:
        raise AssertionError("unrelated set must contain 100 unique items")
    if len(arrays) != 70 or len({case["id"] for case in arrays}) != 70:
        raise AssertionError("array set must contain 70 unique items")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("unrelated-v1.jsonl", unrelated), ("arrays-v1.jsonl", arrays)):
        path = OUT / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(path, len(rows))


if __name__ == "__main__":
    main()
