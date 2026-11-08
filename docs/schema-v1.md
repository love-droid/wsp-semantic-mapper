# Insurance submission mapping v1

The first target intentionally covers information available in the supplied LawPAK application and quote. It is an extraction task, not a coverage recommendation, pricing calculation, or underwriting decision.

## Payload groups

| Group | Meaning |
| --- | --- |
| `insured` | Applicant/named-insured legal name and separately supplied DBA. |
| `addresses` | Addresses of the insured with explicit roles. Excludes the broker/carrier address. |
| `policy_period` | Policy effective and expiration dates. Excludes submission, issue, and quote-valid-until dates. |
| `business` | Attorney count, non-attorney staff count, establishment year, annual payroll. |
| `practice_areas` | Selected practice areas and stated percentage points. Excludes printed but unselected choices. |
| `risk_answers` | Three specific answered questions, encoded as true/false/null. |
| `coverage_terms` | Numeric limits, deductibles, and retentions in account-specific schedules, preserving their coverage, component, and basis. |
| `annual_premium` | Explicit total annual policy premium and explicitly identified currency. |

## Annotation rules

1. Every defined object key is present. Unsupported scalar values and unresolved conflicting values are `null`, not an empty string, zero, false, or an invented default. Record conflicts in annotation metadata; do not invent a source-precedence rule after seeing a model prediction.
2. A list is `null` when its contents are not established by the document. `[]` requires explicit evidence that the list contains no items.
3. Process one source document per initial case. Do not fill a quote's missing attorney count from its companion application. Multi-document mapping is a later task with explicit input scope.
4. Preserve distinctions between legal name, DBA, contact, broker, and carrier. A primary office is not automatically a mailing address.
5. Format dates as `YYYY-MM-DD` only after establishing the source date convention. A policy expiration date differs from quotation validity.
6. Amounts are numbers without separators. Counts are integers. Postal codes remain strings. Percentages use percentage points, not fractions. No currency conversion or unstated calculations.
7. Currency requires an explicit code/name or a declaration applying to the relevant section. `$` alone is ambiguous. In the LawPAK quote seed, the premium header explicitly identifies USD; the coverage table does not, so those row currencies are `null`.
8. Carry table headings or merged cells forward only within the relevant section. Retain each row's basis. A combined `Each Occurrence / Aggregate` basis remains one row, without inferring two independent amounts.
9. `coverage_terms` covers numeric limits, deductibles, and retentions only. Statutory wording, unresolved `[Scheduled]`/`[Retro Date]` template fields, and rows without numeric amounts are outside this first list. This payload is not a complete representation of all policy terms.
10. Preserve source wording for practice areas, coverage names, components, and bases. Trim surrounding whitespace and normalize runs of whitespace; substantive labels and punctuation remain unchanged.
11. Evidence is stored alongside the expected payload as page references and transcription notes, not inside the model's target output. Do not include the expected payload or annotation notes in the model prompt.

## Seed examples and review status

The ignored `data/local/development.seed.jsonl` contains two examples from one insured, using manually aligned source text checked against rendered PDF pages. These are curated semantic-mapping inputs, not raw PDF-extraction outputs. Their review status records that they have not been reviewed by the user.

Both have the same `group_id`. They are development examples; neither is an independent holdout. Future splits must keep the same insured, companion application/quote, near-duplicates, and derived variants together. Six source PDFs do not become forty independent documents by splitting pages or paraphrasing questions.

The quote's rendered page 1 and letter body identify the law firm, but the rendered page 2 `NAMED INSURED` cell reads `LawPak`. Under the explicit conflict rule, this seed's `insured.legal_name` is `null` pending resolution. The model input preserves this inconsistency. Plain text extraction alone interleaves these labels and values and can conceal the error.

Sources are preserved under ignored `data/local/raw/` with SHA-256 digests in `data/local/source_manifest.json`. The quoted evidence should be rechecked if source hashes change. No private source document needs to be committed to reproduce the public schema definition.

## First comparison

Start with the insured identity and policy-period fields as a short-output task, then expand to practice-area and coverage arrays. Freeze the chosen output projection for each original-versus-edited comparison and report the exact fields scored.

Score field values and exact payload agreement separately from JSON parsing and schema validity. Distinguish `null`, `false`, and zero. Score row contents and duplicates, not just list length. Use a format checker when validating schema dates; checking only the date string's shape is insufficient.

Schema validation cannot establish semantic correctness, identify the correct person, verify a monetary basis, or prove that a source supports an answer. Those checks need the annotated reference values. No performance improvement is claimed at this stage.
