# Part Classification — Technical Documentation

> Module for individual analysis of a technical CAD drawing (PDF): part classification, cited standards, GD&T frame detection (feature control frames), and datum detection, with compliance evaluation against ISO 1101 / ISO 5459 as a reference baseline.

This document explains **how each part of the pipeline works**, in the order data flows through it — same format as `CAD_REVIEW_TECHNICAL_DOCUMENTATION.md`, meant to serve as a presentation script. Each section can become a block of slides, with source code and design decisions already justified.

---

## 1. Overview of the flow

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌───────────────────┐
│  Upload of   │→  │  Vector text │→  │  Part             │→  │  GD&T + Datum     │→  │  Parsing +          │→  │  Final aggregation  │
│  1 PDF       │   │  extraction  │   │  classification    │   │  detection        │   │  ISO evaluation     │   │  + visual evidence  │
│              │   │              │   │  (LLM)            │   │                   │   │                     │   │                     │
└──────────────┘   └──────────────┘   └──────────────────┘   └──────────────────┘   └────────────────────┘   └───────────────────┘
```

Input: **one** PDF file (the part to be analyzed, unlike CAD Review, which compares two).
Output: part classification with textual evidence, standards cited in the drawing, detected GD&T frames with tolerance/datums read from them, datums defined in the drawing, ISO 1101/ISO 5459 compliance findings, and annotated images (per page) ready for download.

The entire flow is implemented in `pages/classification.py` (Streamlit interface) and delegates the heavy processing to **a single deterministic entry point + 1 LLM call**: `process_cad_pdf()`, in `src/cad_review/folder_pipeline.py`.

| Module | Responsibility |
|---|---|
| `src/cad_review/folder_pipeline.py` | Orchestrates the entire pipeline: calls classification, GD&T detection, parsing, ISO checks, generates artifacts and writes `result.json` |
| `src/cad_review/orchestrator.py` | LLM classification call + deterministic applicable-standards lookup |
| `src/cad_review/compliance_engine.py` | Aggregates everything into a single result, generates the `findings` list and the `summary` |
| `src/gdt/detector.py` | Geometric detection of GD&T frames (feature control frames) from the vector PDF |
| `src/gdt/symbol_classifier.py` | Ranking-based classification of the first cell's symbol by comparison against a template catalog |
| `src/gdt/frame_parser.py` | Structural reading of the frame: numeric tolerance, diameter, referenced datums |
| `src/gdt/datum_feature.py` | Detection of datum indicators (box + filled triangle) in the drawing |
| `src/gdt/iso1101.py` / `src/gdt/iso1101_reference.py` | ISO 1101 edition resolution and evaluation of the datum requirement per characteristic |
| `src/gdt/datum_consistency.py` | Checks whether each referenced datum has a matching definition (ISO 5459) |
| `src/cad_review/visual_output.py` | Generates the annotated images (GD&T, datums, combined) per page |
| `src/utils/standards_applicability.py` | Deterministic lookup of applicable standards via `Normas.xlsx` |

---

## 2. Upload and preview

**Where:** `pages/classification.py`.

```python
pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_classification")
```

Just like in CAD Review, the first page is rendered at low resolution (100 DPI, via `pdf_to_pil_images`) purely for visual confirmation before the heavy processing — the same "cheap preview before the real cost" logic described in the CAD Review document.

Before triggering the analysis, the page validates that the three mandatory pipeline configuration files exist (`Normas.xlsx`, `assets/gdt/templates/`, `validation/gdt/configs/iso1101_2017_reference_rules.json`). If any is missing, the UI shows an explicit error instead of letting the pipeline fail with a generic `FileNotFoundError` mid-processing.

---

## 3. Extracting vector text from the PDF

**Where:** `src/modeling/llm_models.py` → `extract_text_from_pdf()`, called inside `run_part_classification_branch()`.

Unlike CAD Review (which sends **images** to the model), part classification is done from the PDF's **vector text** — extracted directly with PyMuPDF (`page.get_text()`), with no rasterization or OCR. This is possible because the information classification needs (part name, material, cited standards) is normally found in real text blocks of the PDF (title block, notes), not in purely graphical drawing content.

Extracting vector text instead of sending an image to the LLM is a deliberate cost choice: text is orders of magnitude cheaper in tokens than an image, and for this specific problem (reading fields from a title block) there is no gain in using the model's computer vision.

---

## 4. Part classification by AI (LLM)

**Where:** `src/cad_review/orchestrator.py` → `run_part_classification_branch()`. Actual model call in `src/modeling/llm_models.py` → `classify_cad_enriched()`. Prompt in `prompts.py` → `classificacao_enriquecida_prompt`.

This is the **only LLM call** in the entire Part Classification pipeline — everything else (GD&T, datums, ISO) is 100% deterministic. The text extracted in the previous step is injected into the prompt (placeholder `{{texto_extraido}}`) and sent to Gemini with **mandatory structured output** (`response_schema=CadClassificationEnriched`), guaranteeing the response is always a JSON payload validated by Pydantic, never free text.

### 4.1 What is extracted

| Field | Description |
|---|---|
| `document_type` | Document type (`product_drawing`, `assembly_drawing`, `process_sheet`, `technical_specification`) |
| `component` | Part name (e.g., "Connecting Rod"), read from the PART NAME/DESCRIPTION field or title |
| `material_family` | Material family (e.g., `sintered_metal`, `gray_cast_iron`) |
| `compressor_series` | Compressor series — **only when there is explicit mention** in the text |
| `cited_standards` | List of standards literally cited in the text (with the evidence snippet) |

Every field (except `cited_standards`) is returned as an `{value, evidence, confidence}` object — never just a bare value. This is the same "answer with evidence, not with assertion" pattern used throughout the rest of the project.

### 4.2 Maximum-precision prompt rules

The prompt is explicit in prohibiting invention: if a field is not written in the text, the model must return `null` instead of guessing. The strictest rule concerns `compressor_series` — the prompt reinforces three times that the series must **never** be inferred from a part code or other fields, and only accepts a literal mention (`"SERIES F"`, `"SÉRIE EG"`). This exists because the compressor series determines which set of standards is applicable (section 5), so a made-up value would silently propagate an error further down the pipeline.

Likewise, standard codes must be preserved **exactly as they appear** in the text (`"TSS002611"`, `"TSS-002611"`, `"TSS 002611"` remain distinct at this stage — normalization happens later, in section 5).

### 4.3 Temporary compressor-series policy

**Where:** `run_part_classification_branch()`, constant `DEFAULT_COMPRESSOR_SERIES_CONTEXT = "ALL"`.

The real compressor series should come from an external system (Windchill), which is not yet integrated. In the meantime, the pipeline uses `"ALL"` as the review context (`review_context.compressor_series`) — deliberately **separate** from the `compressor_series` field the LLM extracted from the CAD itself (which remains whatever was actually read from the drawing, normally `None`). This prevents the code from conflating "I don't know the series" with "the series is ALL."

---

## 5. Standards: deterministic applicability and comparison

**Where:** `src/utils/standards_applicability.py` → `StandardsApplicabilityEngine`, `compare_standards()`. Orchestrated in `run_part_classification_branch()`.

After the LLM extracts `component` and the cited standards, a second, **fully deterministic** step (no LLM) queries the `Normas.xlsx` spreadsheet to find out which standards *should* be present in the drawing:

1. **Fuzzy match of the component** (Jaccard similarity over normalized tokens) against the `Parts` sheet → mandatory standards for that part type.
2. **Standards from the `Notes` sheet** whose `Applicability` contains the component (or is `"All"`) and whose `Compressor_Series` is compatible with the current context.
3. **Match by material family** (if provided) against standards in the `Material` category.

The result is deterministically compared against the standards cited by the LLM via `compare_standards()`:

```python
matching   = expected ∩ cited
missing    = expected - cited
unexpected = cited - expected
```

> **Note on the temporary "ALL series" policy:** since the real series is not yet known, the pipeline keeps only the applicability rows originating from the `component_match` path (`Notes` sheet, which already filters by series) and **discards** the aggregated rows from the `Parts` sheet (which historically mix standards from all series into a single per-part list) — avoiding declaring a standard applicable when it actually only applies to a different series than the real one.

This comparison block (`standards_comparison`, `applicable_standards`) is still computed and persisted in `result.json` even though the current Streamlit interface does not display it as a dedicated tab — it feeds the `"standards"`-domain *findings* (section 8) and remains available in the **Full JSON** tab.

---

## 6. Detecting datums defined in the drawing

**Where:** `src/gdt/datum_feature.py` → `detect_datum_feature_indicators()`. Called in `process_cad_pdf()` **before** GD&T detection, in a first pass over all pages.

A datum is considered "defined" in the drawing only when three independent signals coincide — a lone letter (e.g., just the text `"A"`) is **not** enough:

1. a PDF text token that is exactly one uppercase letter (`^[A-Z]$`);
2. a small, near-square rectangular outline enclosing that text (detected via OpenCV over the rasterized, binarized page);
3. a filled triangular marker nearby, connected to the box by a continuous "corridor" of ink (`stem_coverage`) — the stroke that links the datum's flag to the triangular arrow.

```python
def detect_datum_feature_indicators(
    pdf_bytes: bytes, *, page_index: int = 0, raster_dpi: int = 200,
    min_box_size_pt: float = 7.0, max_box_size_pt: float = 24.0, ...
) -> list[DatumFeatureIndicatorCandidate]:
```

This requirement of 3 simultaneous signals exists to avoid confusing any random letter on the drawing (a dimension, a note, a revision) with an actual datum. Each accepted candidate stores `marker_side` (which side of the box the marker is on), `stem_coverage`, and `box_rectangularity` — geometric confidence metrics, not a calibrated probability.

The search runs **across all pages before evaluating any reference**, because a datum referenced in a GD&T frame on page 1 may be defined on page 2 (e.g., an auxiliary view on another sheet) — evaluating page by page in isolation would miss that case.

---

## 7. Geometric detection of GD&T frames

**Where:** `src/gdt/detector.py` (subclass) and `src/utils/gdt_detector.py` (base implementation). Called in `process_cad_pdf()` via `GdtFrameDetector().detect_frames()`.

A "feature control frame" (the rectangular GD&T box, with the symbol in the first cell followed by tolerance and datums) is reconstructed **from the PDF's vector line segments** — not from the rasterized image. The detector scans the page's horizontal/vertical segments and groups the ones that form a rectangle subdivided into cells, within size/proportion ranges plausible for an FCF.

The class in `src/gdt/detector.py` reuses the `frame_bbox` (the outer rectangle) exactly as it came from the legacy implementation in `src/utils/gdt_detector.py`, but re-segments the **internal cell dividers** with a stricter tolerance — this prevents the symbol's own stroke (e.g., the vertical bar of the Position symbol ⌖) from being mistaken for a cell divider line.

Each candidate (`GdtFrameCandidate`) has an explicit `candidate_id` in the format `GDT-CAND-P<page>-<sequential>` — the `CAND` prefix is not decorative: it reinforces throughout the whole chain (UI, report, diagnostic CSV) that **this is a detector candidate, not a GD&T frame validated by a human**, until someone confirms it.

---

## 8. Symbol classification (ranking against a template catalog)

**Where:** `src/gdt/symbol_classifier.py` → `load_template_catalog()`, `render_page_gray()`, `score_candidates()`.

After detecting the frame's geometry, the system tries to identify **which GD&T symbol is in the first cell** (parallelism, position, flatness, etc.) by comparing that cell's crop against a catalog of reference images at `assets/gdt/templates/<class>/`.

This is **purely deterministic and LLM-free** — no text or vision AI call at all. The algorithm:

1. Crops the interior of the symbol cell (`crop_cell_interior`), removing the border so as not to compare line against line.
2. Normalizes contrast/polarity and generates three representations (`gray`, `binary`, `edges`) of the crop.
3. Compares against each template using template correlation (`cv2.matchTemplate`) **and** two global shape descriptors: a structural occupancy/projection descriptor and a HOG (Histogram of Oriented Gradients) over a 3×3 grid.
4. Combines the five scores into two families — local appearance (40%) and global shape (60%) — avoiding three variants of the same local evidence (`gray`/`binary`/`edges`) being counted as three independent votes.
5. Returns the best-scoring class, the second best, and the margin between them (`CandidateSymbolScore`).

> **This is ranking, not threshold classification.** There is no calibrated acceptance cutoff ("if score > X, it is this symbol"). The result is always "given this catalog, this is the best match and by how much it beat the second-best" — the final decision to accept it or not is left to whoever reads the result (a human reviewer or a downstream rule).

If the template catalog is incomplete (`symbol_catalog.complete == False`) and `allow_incomplete_symbol_catalog=False` (the default), this step is **entirely disabled** for the run — no candidate receives a characteristic classification, in a fail-closed way (silent, no crash, but explicit in the result).

---

## 9. Structural parsing of the GD&T frame

**Where:** `src/gdt/frame_parser.py` → `parse_feature_control_frame()`.

With the geometry and (when available) the classified characteristic, this step reads the **content** of each frame cell, text by text extracted from the vector PDF — with no visual inference in this version of the pipeline (the `visual_evidence` parameter exists in the signature as a visual fallback, but `process_cad_pdf()` does not populate it today):

- **Tolerance cell** (index 1): extracts the first recognizable decimal number (`_extract_first_number`), accepting both `,` and `.` as decimal separators, and detects whether a diameter symbol (`⌀`, `Ø`, `∅`) is textually present.
- **Datum cells** (index 2+): a cell is accepted as a datum **only if it contains exactly a single token that is a lone uppercase letter** (`_extract_structural_datum`) — anything else in the cell (multiple tokens, a number alongside it, a modifier) makes the content fall into `unresolved_tokens` instead of being guessed.

The result (`ParsedGdtFrame`) explicitly documents **where** each field came from (`field_sources`) and what **was not** resolved (`unresolved_fields`, `unresolved_tokens`) — there is no `tolerance_value` field filled in with a guess; if the text does not allow a number to be extracted, the field stays `None` and the gap shows up in the unresolved list.

---

## 10. ISO 1101 evaluation (datum requirement per characteristic)

**Where:** `src/gdt/iso1101.py` (rules/edition) + `src/gdt/iso1101_reference.py` (user-facing finding). Called in `process_cad_pdf()` via `assess_iso1101_datum_rule()`.

This is the core of the "compliance" logic for GD&T: each geometric characteristic (parallelism, flatness, position, etc.) has a different datum requirement under ISO 1101:2017 — some **require** a reference datum, others **use none**, others are **conditional** (depends on design context, cannot be decided by presence/absence alone).

The rule table comes from `validation/gdt/configs/iso1101_2017_reference_rules.json` and is cited literally in the finding (`source_ref`, e.g., `"ISO 1101:2017 Table 1, subclause 18.9"`) — no rule is hard-coded in Python; everything comes from external, traceable configuration.

### 10.1 The four possible outcomes

| Rule's `datum_requirement` | Situation found | Result |
|---|---|---|
| `required` | No referenced datum | 🟡 `WARNING` — "Potential violation": requires a datum and has none |
| `required` / `none` | Presence/absence matches the rule | 🟢 `PASS` |
| `none` | Datum referenced anyway | 🟡 `WARNING` — datum present where the rule says it shouldn't be |
| `conditional` | Any situation | 🔎 `NEEDS_CONTEXT` — datum presence/absence alone decides nothing; needs design context |

### 10.2 `reference` vs `normative` mode

The finding is generated with `mode="reference"` in the current pipeline — meaning the wording is always **"Potential violation of ISO 1101:2017"**, never a categorical non-compliance statement (`normative_claim=False`). This is intentional: the tool uses ISO 1101:2017 as a **technical reference baseline** to raise points of attention, not as proof that that edition of the standard is contractually applicable to the drawing — that normative determination (exact edition, actual applicability) is out of scope for this version of the pipeline (see `src/gdt/iso1101.py::resolve_iso1101_edition`, which exists precisely to never *assume* an edition without an explicit citation or a supplied applicability rule).

---

## 11. ISO 5459 evaluation (datum definition consistency)

**Where:** `src/gdt/datum_consistency.py` → `assess_referenced_datum_definitions()`. Called in `process_cad_pdf()` for each processed GD&T frame.

While section 10 asks "should this characteristic have a datum?", this step asks "does the datum the frame references **actually exist** in the drawing?" — cross-referencing `referenced_datums` (extracted during parsing, section 9) against `datum_definitions` (detected in section 6):

```python
if datum in definitions:
    status = "PASS"       # ISO5459_DATUM_DEFINITION_FOUND
else:
    status = "WARNING"     # ISO5459_REFERENCED_DATUM_NOT_DEFINED — "Potential violation of ISO 5459"
```

Each datum is evaluated **only once per frame** even if it appears repeated (`seen: set[str]`), and tokens that are not a single uppercase letter are silently ignored (protection against noise from earlier parsing). Just as in section 10, the default mode is `reference` — the wording never categorically asserts a violation, only flags the point for human review.

---

## 12. Final aggregation: findings and summary

**Where:** `src/cad_review/compliance_engine.py` → `build_cad_review_result()`. Contracts in `src/cad_review/types.py`.

This step **does not call an LLM and does not reinterpret anything** — it only normalizes the results from sections 5, 10, and 11 into a single list of `CadReviewFinding`, each one with:

```python
{
  "finding_id": "F-001",
  "domain": "standards" | "iso1101" | "iso5459",
  "status": "PASS" | "WARNING" | "NEEDS_CONTEXT" | "NOT_EVALUATED",
  "severity": "INFO" | "WARNING",   # ERROR is never emitted by the current logic
  "code": "...",                     # e.g., ISO1101_REQUIRED_DATUM_MISSING
  "finding": "human-readable result text",
  "recommended_action": "...",
  "candidate_id": "...",             # references the GD&T frame, when applicable
  "datum": "...",                    # references the datum letter, when applicable
  "normative_claim": false,
}
```

The `summary` is just a per-status count (`PASS`/`WARNING`/`NEEDS_CONTEXT`/`NOT_EVALUATED`) over that list — there is no aggregated "overall compliance" score or grade; the reading is always item by item.

The final `gdt_frames` in `result.json` is **not** the reduced format that `build_cad_review_result()` builds internally (which only keeps characteristic/tolerance/datums) — `process_cad_pdf()` deliberately overwrites that field with the rich version (`raw_frames`, full geometry + complete `symbol_scoring`), because the interface and the visual report need the geometry (bounding boxes) to draw the annotations.

---

## 13. Visual evidence (annotated images per page)

**Where:** `src/cad_review/visual_output.py` → `render_visual_evidence()`. Called at the end of `process_cad_pdf()`.

For each page of the PDF, three PNG images are generated and saved to disk (not in memory — the UI reads the file from disk afterward):

| Image | Content | Used in the UI |
|---|---|---|
| `page_NNN_annotated.png` | GD&T + datums combined in the same image | **Marked Drawing** tab |
| `page_NNN_gdt.png` | GD&T frames only | **GD&T Evaluation** tab |
| `page_NNN_datums.png` | Datums only | **Datum Definitions** tab |

The color of each drawn rectangle reflects the **status of that candidate's/datum's finding** (green = PASS, red = WARNING, orange = NEEDS_CONTEXT, blue = NOT_EVALUATED) — i.e., the image already visually embeds the conclusion of the ISO evaluation; it is not a neutral overlay. Labels are placed in "free lanes" around the source geometry (`_find_free_label_position`), with a line connecting the label back to the rectangle, to reduce overlap on dense drawings.

Besides the three images per page, individual crops of each frame/datum (`crops/GDT-CAND-P01-001_frame.png`, `crops/DATUM-A_001_01.png`) are saved separately — not shown today in the simplified UI, but available in the working directory and referenced in `artifacts.visual_evidence.crops`.

---

## 14. Detection diagnostics (not exposed in the current UI)

**Where:** `src/cad_review/detection_diagnostics.py` → `render_detection_diagnostics()`. Also called at the end of `process_cad_pdf()`.

Alongside the "final" visual evidence, the pipeline generates a second set of artifacts intended for whoever is **validating the detector itself**, not for the tool's end user:

- `page_NNN_candidates.png` — only the Phase-1 geometry candidates (before any classification), labeled solely with the candidate ID.
- `page_NNN_symbol_contact_sheet.png` — one card per candidate, showing the frame crop, the symbol crop, and the full ranking (top-3 classes with score) returned by the classifier.
- `candidate_diagnostics.csv` — one row per candidate with geometry, ranking, and **three intentionally blank columns** (`human_is_real_gdt`, `human_true_characteristic`, `human_notes`) for an engineer to fill in manually during validation.

This separation exists so that two questions can be answered independently: "did Phase 1 (geometry) propose the right frame?" and, if so, "did Phase 2 (symbol) classify it correctly?" — without one artifact mixing the two layers of uncertainty.

---

## 15. Displaying results in the interface

**Where:** `pages/classification.py`.

At the end of processing, the result is organized into tabs (the UI was deliberately simplified to focus on visual evidence + download, not detailed findings tables):

1. **🏷️ Classification** — component, material family, document type, and compressor series (each with value + textual evidence), and the list of standards cited in the drawing **filtering out ISO standards** (which are handled separately by the GD&T/ISO logic, not treated as a cited "part standard").
2. **🖼️ Marked Drawing** — the combined (GD&T + datums) image per page, with a PNG download button.
3. **📐 GD&T Evaluation** — the GD&T-only image per page, with a download button.
4. **🎯 Datum Definitions** — the datums-only image per page, with a download button.
5. **🗂️ Full JSON** — the full `result.json`, browsable on screen and available for download.

Note that the underlying `result.json` contains far more detail than the UI exposes directly (standards comparison, each individual ISO 1101/5459 finding, the symbol classifier's full scoring, each candidate's geometry) — all of that remains accessible via the **Full JSON** tab for anyone who needs a full audit, even though the main experience prioritizes the marked images.

Each analysis runs in its own temporary directory (`tempfile.mkdtemp(prefix="cad_review_")`), automatically cleaned up before a new analysis starts.

---

## 16. Module summary and where to find each piece of code

| Pipeline step | File | Function/Snippet |
|---|---|---|
| Upload and preview | `pages/classification.py` | `PDF Upload` section |
| Vector text extraction | `src/modeling/llm_models.py` | `extract_text_from_pdf` |
| Part classification (LLM) | `src/modeling/llm_models.py` | `classify_cad_enriched` |
| Classification prompt | `prompts.py` | `classificacao_enriquecida_prompt` |
| Classification + standards orchestration | `src/cad_review/orchestrator.py` | `run_part_classification_branch` |
| Applicable standards lookup | `src/utils/standards_applicability.py` | `StandardsApplicabilityEngine.get_applicable_standards` |
| Deterministic standards comparison | `src/utils/standards_applicability.py` | `compare_standards` |
| Defined-datum detection | `src/gdt/datum_feature.py` | `detect_datum_feature_indicators` |
| Geometric GD&T frame detection | `src/gdt/detector.py`, `src/utils/gdt_detector.py` | `GdtFrameDetector.detect_frames` |
| Symbol classification (ranking) | `src/gdt/symbol_classifier.py` | `load_template_catalog`, `score_candidates` |
| Structural frame parsing | `src/gdt/frame_parser.py` | `parse_feature_control_frame` |
| ISO 1101 edition resolution | `src/gdt/iso1101.py` | `resolve_iso1101_edition`, `assess_datum_reference_semantics` |
| ISO 1101 finding (datum requirement) | `src/gdt/iso1101_reference.py` | `assess_iso1101_datum_rule` |
| ISO 5459 finding (datum consistency) | `src/gdt/datum_consistency.py` | `assess_referenced_datum_definitions` |
| Final aggregation (findings + summary) | `src/cad_review/compliance_engine.py` | `build_cad_review_result` |
| Orchestration of everything + result.json write | `src/cad_review/folder_pipeline.py` | `process_cad_pdf` |
| Annotated visual evidence | `src/cad_review/visual_output.py` | `render_visual_evidence` |
| Detection diagnostics (CSV + contact sheet) | `src/cad_review/detection_diagnostics.py` | `render_detection_diagnostics` |
| Interface display | `pages/classification.py` | `_render_classification_tab`, `_render_combined_overview`, `_render_gdt_tab`, `_render_datums_tab` |

---

## 17. Presentation highlights

1. **A single AI call across the whole pipeline** — only part classification (section 4) uses an LLM. All the GD&T, datum, and ISO compliance logic (sections 6 through 12) is 100% deterministic: same input, same output, always — important when arguing for auditability and predictable cost.
2. **"Candidate" is a deliberately chosen word** — the `GDT-CAND-` prefix and the "Potential violation" wording in every finding (sections 10 and 11) exist because the pipeline never claims to have validated a GD&T frame or a normative violation against a human; it points, with evidence, to where to look.
3. **Fail-closed, not fail-silent** — when context is missing to decide something (ISO edition not cited, incomplete symbol catalog, conditional characteristic), the system explicitly returns `NOT_EVALUATED`/`NEEDS_CONTEXT` instead of risking a guess. This design choice repeats across at least four different modules (`iso1101.py`, `frame_parser.py`, `symbol_classifier` via `folder_pipeline.py`, `datum_consistency.py`).
4. **Two layers of visual evidence, deliberately separated** — the "final" image (section 13) mixes geometry with the evaluation's result; the diagnostics (section 14) show only the raw geometry, before any judgment. This allows the detector and the compliance logic to be validated independently.
5. **The business rule lives in configuration, not in code** — the per-characteristic datum-requirement table (section 10) is an external JSON with the exact citation of the standard's excerpt; switching editions or adjusting a rule does not require changing Python code.
