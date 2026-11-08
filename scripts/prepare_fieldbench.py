"""Download and freeze FieldBench insurance development/held-out partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
import yaml


DATASET_ID = "fieldbench/corpus"
DATASET_REVISION = "7247f4204e4a60a80875d30ef06ea54f8006304a"
SCHEMA_REVISION = "57cdb82efaa6c0ba856679a56cf63e614fa1959b"
CATEGORIES = {
    "insurance_policies": "policy_declarations.yaml",
    "insurance_claims": "claim_form.yaml",
}
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/fieldbench-v0.4.2"
SCHEMA_DIR = OUTPUT / "schemas"
SPLIT_SALT = "wsp-fieldbench-v0.4.2-split-v1"


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def schema_to_json_schema(schema_doc: dict) -> dict:
    properties = {}
    required = []
    for name, spec in schema_doc["fields"].items():
        field_type = spec["type"]
        json_type = {"string": "string", "date": "string", "number": "number", "integer": "integer", "boolean": "boolean"}[field_type]
        description = spec.get("description", "")
        if spec.get("extraction_hint"):
            description += "\nExtraction hint: " + spec["extraction_hint"].strip()
        mappings = spec.get("mappings") or {}
        if mappings:
            description += "\nCanonical label mappings: " + json.dumps(mappings, ensure_ascii=False, sort_keys=True)
        if spec.get("verbatim"):
            description += "\nExtract verbatim; do not paraphrase."
        prop = {"type": [json_type, "null"], "description": description.strip()}
        if field_type == "date":
            prop["format"] = "date"
        if spec.get("options"):
            prop["enum"] = [*spec["options"], None]
        properties[name] = prop
        required.append(name)
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties}


def split_key(row: dict) -> tuple[str, str]:
    return row["category"], row["source"]


def assign_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata[split_key(row)].append(row)

    development, heldout = [], []
    for stratum_rows in strata.values():
        ordered = sorted(
            stratum_rows,
            key=lambda row: hashlib.sha256(f"{SPLIT_SALT}:{row['doc_id']}".encode()).hexdigest(),
        )
        holdout_count = max(1, round(len(ordered) * 0.2)) if len(ordered) >= 5 else 0
        heldout.extend(ordered[:holdout_count])
        development.extend(ordered[holdout_count:])
    return sorted(development, key=lambda row: row["doc_id"]), sorted(heldout, key=lambda row: row["doc_id"])


def convert_row(row: dict, json_schema: dict) -> dict:
    expected = json.loads(row["expected"])
    return {
        "id": row["doc_id"],
        "group_id": row["doc_id"],
        "category": row["category"],
        "source": row["source"],
        "original_format": row["original_format"],
        "schema_name": row["schema"],
        "input": row["document"],
        "expected": expected,
        "json_schema": json_schema,
        "human_reviewed": False,
        "annotation_source": "FieldBench v0.4.2 released ground truth; external annotation quality is not independently re-reviewed in this project",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    schemas = {}
    for category, filename in CATEGORIES.items():
        destination = SCHEMA_DIR / filename
        if not destination.is_file():
            raise SystemExit(f"Fetch the pinned schema into {destination} before preparing the split")
        schema_doc = yaml.safe_load(destination.read_text(encoding="utf-8"))
        schemas[category] = schema_to_json_schema(schema_doc)

    dataset = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)
    selected = [row for row in dataset if row["category"] in CATEGORIES]
    development, heldout = assign_split(selected)
    dev_cases = [convert_row(row, schemas[row["category"]]) for row in development]
    holdout_cases = [convert_row(row, schemas[row["category"]]) for row in heldout]
    write_jsonl(OUTPUT / "development.jsonl", dev_cases)
    write_jsonl(OUTPUT / "heldout.jsonl", holdout_cases)

    ids = {"development": [row["doc_id"] for row in development], "heldout": [row["doc_id"] for row in heldout]}
    split_manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "schema_source": "https://github.com/fieldbench/corpus",
        "schema_revision": SCHEMA_REVISION,
        "schema_files": CATEGORIES,
        "selected_categories": sorted(CATEGORIES),
        "split_method": "within each category x source, sort by SHA256(salt + doc_id); first round(20%) assigned to heldout; salt is fixed in script",
        "split_salt": SPLIT_SALT,
        "counts": {
            "total": len(selected),
            "development": len(development),
            "heldout": len(heldout),
            "development_by_category_source": dict(Counter(f"{row['category']}:{row['source']}" for row in development)),
            "heldout_by_category_source": dict(Counter(f"{row['category']}:{row['source']}" for row in heldout)),
        },
        "development_jsonl_sha256": hashlib.sha256((OUTPUT / "development.jsonl").read_bytes()).hexdigest(),
        "heldout_jsonl_sha256": hashlib.sha256((OUTPUT / "heldout.jsonl").read_bytes()).hexdigest(),
        "document_ids": ids,
        "license_note": "FieldBench ground truth/schema/tooling carry dataset terms; document rights are per source. Selected synthetic source items are CC0, government-form claim documents are public domain; always retain attribution/provenance metadata.",
    }
    (OUTPUT / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(split_manifest["counts"], ensure_ascii=False, indent=2))
    print(f"Manifest: {OUTPUT / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
