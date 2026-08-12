# CAD Review — Technical Documentation

> Module for automatic comparison between two revisions of a technical CAD drawing (PDF), with visual difference detection and engineering analysis via generative AI (Gemini / GCP Vertex AI).

This document explains **how each part of the pipeline works**, in the order data flows through it. The idea is to serve as a presentation script: each section below can become a slide (or block of slides), with the source code and design decisions already justified.

---

## 1. Overview of the flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Upload the  │ --> │  Rasterizing  │ --> │  Pre-filter of   │ --> │  AI analysis      │ --> │  Visual annotation │
│  2 PDFs      │     │  (PDF→image) │     │  changed pages   │     │  (Gemini/Vertex)  │     │  + PDF report      │
└──────────────┘     └──────────────┘     └──────────────────┘     └──────────────────┘     └───────────────────┘
```

Input: two PDF files (**original** version and **revised** version of the same drawing).
Output: for each page with a difference, a technical table (Item / Difference / Location / Status / Recommended Action), the visual diff image, the image with marked quadrants, and downloadable PDF reports.

The entire flow is implemented in `front.py` (Streamlit interface) and relies on four utility modules:

| Module | Responsibility |
|---|---|
| `src/utils/helper_func.py` | PDF → image conversion, visual diff computation (OpenCV), image compression |
| `src/modeling/llm_models.py` | Call to the Gemini model via Vertex AI (`compare_cad_pages`) |
| `src/utils/cad_quadrant_paint.py` | Extraction of the PDF's zoning grid and painting of the quadrants reported by the AI |
| `src/utils/cost_logger.py` | Logging of tokens, latency, and estimated cost for each LLM call |

---

## 2. Upload and preview

**Where:** `front.py`, `CAD Review Mode` section.

The user uploads two PDF files:

- **Original** — previous revision of the drawing.
- **Revised** — current revision.

```python
pdf1 = st.file_uploader("Upload original PDF", type=["pdf"], key="pdf1")
pdf2 = st.file_uploader("Upload revised PDF", type=["pdf"], key="pdf2")
```

When either file is uploaded, its first page is rendered at low resolution (100 DPI) just so the user can visually confirm they uploaded the right file before triggering the heavy processing. This is a deliberate UX choice: **a fast, cheap preview before the expensive processing**.

---

## 3. Rasterizing the PDFs

**Where:** `src/utils/helper_func.py` → `pdf_to_pil_images()` and `pdf_to_images_base64()`.

PDFs are vector documents; to compare them visually and to send them to the vision model, each page is converted into a raster image using **PyMuPDF (fitz)**:

```python
matrix = fitz.Matrix(dpi / 72, dpi / 72)   # 72 pt = 1 inch (PDF's native unit)
pix = page.get_pixmap(matrix=matrix)
```

The system rasterizes **three times, at three different resolutions**, each optimized for a specific use:

| Resolution | Use | Why |
|---|---|---|
| 200 DPI | Sent to the LLM (`pages*_b64`) | Enough for the model to read text and symbols; keeps the payload smaller |
| 300 DPI | Visual diff and on-screen display (`pages*_pil`) | Sharpness needed to detect small changes via OpenCV and for the user to be able to zoom |
| 150 DPI | Per-ID detail blocks in the PDF report (`pages*_pil_150`) | Intermediate resolution: legible, but avoids generating a huge report PDF |

This resolution scaling per purpose is a direct optimization of cost (tokens) and performance (file size).

---

## 4. Image optimization before sending to the LLM

**Where:** `src/utils/helper_func.py` → `compress_png_for_llm()`.

Before sending images to Gemini, they go through PNG compression without losing any relevant content:

```python
img.save(buffered, format="PNG", optimize=True, compress_level=9)
```

This strips metadata and applies maximum PNG compression (which is lossless — it does not affect the AI's ability to read text/symbols), reducing the payload size by ~30–40%. Fewer image bytes → fewer input tokens → lower cost and latency per call.

---

## 5. Pre-filter: which pages actually changed?

**Where:** `src/utils/helper_func.py` → `count_diff_regions()`.

Before spending an LLM call (the most expensive and slowest part of the pipeline), the system decides **which pages are worth analyzing**, using pure computer vision (OpenCV), with no AI involved:

1. Converts both images of the same page (original vs. revised) to grayscale.
2. Computes the absolute pixel-by-pixel difference (`cv2.absdiff`).
3. Applies a light `GaussianBlur` to ignore rendering noise (anti-aliasing, JPEG artifacts).
4. Thresholds the result (`cv2.threshold`) — only what actually changed remains.
5. Closes small gaps with morphology (`MORPH_CLOSE` + `dilate`), without merging distant regions.
6. Counts the resulting contours with a minimum area (> 30px) — this count is `n_regions`.

A page with `n_regions == 0` is discarded: no relevant visual difference, not worth sending to the LLM. This saves calls on large (multi-page) documents where only a minority of pages actually changed.

> **Presentation point:** this is why the system scales well — the AI cost grows only with the number of pages *that actually changed*, not with the total number of pages in the PDF.

---

## 6. Visual diff (the "pink" overlay image)

**Where:** `src/utils/helper_func.py` → `compute_visual_diff()`.

For pages that passed the pre-filter, the system generates an "X-ray" image of the change:

1. Same diff pipeline as the previous step (absdiff → blur → threshold → morphology), but with a slightly more permissive kernel to draw more "generous" rectangles around the change (padding proportional to image size).
2. For each relevant contour, it draws a **rectangle filled with a semi-transparent pink overlay (alpha 0.6)** over the revised image — no hard red borders, so as not to visually clutter the technical drawing.
3. The result is returned as a new PIL image, which is one of the columns shown on screen (the "Differences" column) and is also available for download as a PDF (`⬇️ Download Diff (PDF)`).

This image is **purely geometric** — it carries no semantic information yet (it doesn't know whether the difference is "normal" or a "problem"). That layer of judgment comes in the next step.

---

## 7. Semantic analysis by AI (Gemini via Vertex AI)

**Where:** `src/modeling/llm_models.py` → `compare_cad_pages()`. Prompt in `prompts.py` → `system_prompt`.

This is the core value-adding step of the product. Both page images (original + revised, at 200 DPI) are sent to the **Gemini** model through the `google.genai` SDK, configured to use **GCP Vertex AI** infrastructure:

```python
client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

response = client.models.generate_content(
    model=model,
    contents=[
        types.Part.from_bytes(data=image1_data, mime_type="image/png"),
        types.Part.from_bytes(data=image2_data, mime_type="image/png"),
        types.Part.from_text(text=f"{system_prompt}\n\nFirst image: ORIGINAL. Second image: REVISED."),
    ],
)
```

### 7.1 What the prompt asks for, exactly

The `system_prompt` instructs the model to act as a **senior expert in CAD document analysis** and to report every difference found according to four fixed criteria:

| Column | Content |
|---|---|
| **Difference Found** | Concise description of the change (multiple points separated by `;`) |
| **Location (Quadrant)** | Where the change occurs, referencing the drawing's own coordinate system (e.g., `D4 to E7`, `A1, B1-C3`) |
| **AI Status** | Risk classification into 3 fixed levels (see below) |
| **Recommended Action** | What to do about it |

### 7.2 The three status levels (the heart of the business logic)

- 🟢 **Approved** — the change is correct, intentional, and raises no technical doubt.
- 🟡 **Approved with Observation** — the change is probably intentional, but contains something that deserves human verification (e.g., a change of referenced standard, a new process requirement, a change to an identification symbol).
- 🔴 **Requires Correction** — a clear error, technical inconsistency, or omission.

This 3-level taxonomy is what turns "detecting a difference" into "giving an engineering opinion" — it's the difference between a generic image diff and an actual technical review tool.

### 7.3 Maximum sensitivity rules

The prompt explicitly reinforces that the model must report **every** difference, no matter how small: numeric values (dimensions, tolerances, angles), text that appears/disappears (including inside variant tables), swapping/adding/removing GD&T symbols (naming the symbol, never generically), altered geometry and technical notes. The requirement to "name the symbol" (e.g., "cylindricity symbol ⌭ replaced by circularity symbol ○") exists because a human reviewer needs to know *exactly* what changed, not just that "something" changed.

### 7.4 Output format

The model responds in **plain Markdown**, with a 5-column table (`Item | Difference Found | Location (Quadrant) | AI Status | Recommended Action`). Structured JSON is not requested at this stage — the choice of Markdown makes it easier to render directly on screen (`st.markdown`) and to later parse it for the PDF and for quadrant painting (next sections).

### 7.5 Usage metadata

Each call returns `usage_metadata` (input, output, and total tokens), and latency is measured in code (`time.time()` before/after). This feeds the `CostLogger` (section 10).

---

## 8. Automatic localization of changes on the drawing (quadrants)

**Where:** `src/utils/cad_quadrant_paint.py`.

This is the most sophisticated part of the pipeline and solves a specific problem: **the AI has already said *where* the change is, as free text** (e.g., `"Central (D4 to E7)"`), but this needs to be turned into a **rectangle drawn on the image**. Important: **no additional AI call is made here** — it is 100% deterministic, reusing the text the model already produced in the table.

The process has two independent steps:

### 8.1 Zoning grid extraction (`extract_grid`)

Engineering technical drawings typically have a reference grid along the sheet's edges: numbers along the top/bottom, letters along the sides (like a geographic map). The system reads this grid **directly from the vector PDF** (not from the rasterized image) using PyMuPDF:

1. Scans the text blocks near the page's four edges (`EDGE_FRACTION = 13%` of width/height).
2. Numeric candidates near the horizontal edges → **column** candidates; alphabetic candidates near the vertical edges → **row** candidates.
3. Since the extracted text can contain noise (other numbers/letters that aren't part of the grid), the algorithm (`_best_progression`) looks for the **largest subset of evenly-spaced labels** — i.e., it validates that "1, 2, 3, 4..." really form an arithmetic progression before accepting it as a real grid.
4. The result (`GridInfo`) maps each label (`"D"`, `"4"`) to a real coordinate in PDF points, along with the spacing (`column_step`, `row_step`) between cells.

If the page does not have a grid detectable with confidence (fewer than 3 consistent labels), the function returns `None` — and the pipeline simply doesn't paint quadrants for that page (a safe failure that doesn't break the rest of the analysis).

### 8.2 Parsing free-form location text (`parse_quadrant_text`)

The text the AI writes in the "Location" column is natural language with some structure, not a rigid format. The parser uses regular expressions to recognize three patterns:

- **Single cell**: `"A1"` or `"1A"` (both letter-number and number-letter order).
- **Range**: `"D4 to E7"` or `"D4-E7"` (connected by `"-"` or the word `"to"`).
- **List**: multiple cells separated by commas, each treated as an independent group.

Texts with no recognizable token at all (e.g., `"Central / Detail Views"`) simply generate no group — no error, just no rectangle.

### 8.3 Painting on the image (`paint_quadrants`)

With the grid and cell groups resolved, each rectangle is converted from PDF points to pixels at the rasterization resolution (`bbox_pt_to_px`), and drawn onto the revised image:

- Semi-transparent fill (alpha ~70/255) cropped **only within the rectangle's own region** — no overlay the size of the whole page is allocated, which used to cause `MemoryError` on large sheets (A0/A1) rasterized at high resolution.
- Solid colored border around the rectangle.
- Item number centered inside the quadrant, with a thin white outline to stand out against the drawing's lines.
- The color is chosen by item index from a fixed palette of 6 distinct colors (`_CORES`), ensuring nearby items on the drawing don't get visually confused.

Each `PaintedRegion` stores whether resolution succeeded (`resolvido: bool`) — used on screen to warn the user when "N items could not be located on the grid" (location text with no identifiable quadrant).

### 8.4 Individual per-item painting (`paint_single_item`)

A variant of `paint_quadrants` for drawing **one item at a time** — used in the "Details by ID" section of the PDF report, where each table row gets a pair of images (original + revised) with only that item highlighted, making point-by-point reading easier.

---

## 9. Displaying results in the interface

**Where:** `front.py`, the `Display of results` block.

For each analyzed page, the screen shows:

1. **Image grid** (3 or 4 columns, with zoom): Original | Revised | Differences | *(Revised with Quadrants, if the grid was detected)*.
2. **Download buttons**: Diff as PDF, AI Report as PDF, Revised with Quadrants as PDF.
3. **AI call metrics**: Input Tokens, Output Tokens, Total Tokens, Latency.
4. **Divergence table** rendered as Markdown (`st.markdown`), directly from the model's response.

---

## 10. Generating the technical PDF report

**Where:** `front.py`, inside the "Download AI Report (PDF)" button, using **ReportLab**.

The Markdown text returned by the model is reprocessed to generate a formal PDF document, formatted in A4 landscape, with:

- Title and institutional header (`author="CAD Review - Nidec"`).
- **Manual parsing of the Markdown table** (lines starting and ending with `|`, discarding the `---` separator line), rebuilt as a ReportLab `Table`.
- **Conditional coloring of the AI Status column**: green for "Approved", amber for "Approved with Observation", red for "Requires Correction" — both in the text and the cell background.
- **Automatic bullet points**: cells with `;` are broken into a list with a `•` marker.
- **"Details by ID" section**: for each table row, a dedicated block with the difference description and the two images (original/revised) annotated with only that item, side by side — using exactly the painting functions from section 8.

This report is the final artifact that can be shared with an engineer who has no access to the tool — the "documentary evidence" of the analysis.

---

## 11. Cost and performance tracking

**Where:** `src/utils/cost_logger.py`.

Every LLM call is logged to `custos.csv` (timestamp, model, input/output/total tokens, latency, estimated cost in USD). Cost is calculated using a price-per-million-tokens table (parameterizable according to the model/region used in Vertex AI). This log persists across sessions and serves both operational tracking (how much the tool is costing) and latency benchmarking.

---

## 12. Module summary and where to find each piece of code

| Pipeline step | File | Function/Snippet |
|---|---|---|
| Upload and preview | `front.py` | `PDF Upload` section |
| PDF→image rasterization | `src/utils/helper_func.py` | `pdf_to_pil_images`, `pdf_to_images_base64` |
| Compression for the LLM | `src/utils/helper_func.py` | `compress_png_for_llm` |
| Pre-filter of changed pages | `src/utils/helper_func.py` | `count_diff_regions` |
| Visual diff (pink overlay) | `src/utils/helper_func.py` | `compute_visual_diff` |
| Call to Gemini/Vertex AI | `src/modeling/llm_models.py` | `compare_cad_pages` |
| Model instruction prompt | `prompts.py` | `system_prompt` |
| PDF grid extraction | `src/utils/cad_quadrant_paint.py` | `extract_grid`, `extract_grid_from_page` |
| Location text parsing | `src/utils/cad_quadrant_paint.py` | `parse_quadrant_text` |
| Quadrant painting | `src/utils/cad_quadrant_paint.py` | `paint_quadrants`, `paint_single_item` |
| Markdown table parsing | `src/utils/cad_quadrant_paint.py` | `parse_markdown_table`, `encontrar_coluna` |
| PDF report generation | `front.py` | `Download AI Report (PDF)` block (ReportLab) |
| Cost/latency logging | `src/utils/cost_logger.py` | `CostLogger.log_analysis`, `get_summary` |

---

## 13. Presentation highlights

Suggested "key messages" for each stage, in case you want to build slides from this document:

1. **Layered pipeline** — cheap computer vision filters out what's worth sending to the expensive AI. This is a cost/architecture decision, not just a technical one.
2. **The AI doesn't replace the reviewer, it prioritizes their work** — the 3 status levels (Approved / Observation / Correction) turn the output into a human review priority queue, not a blind automated decision.
3. **Visual traceability** — the quadrant step (section 8) is 100% deterministic and reuses the AI's output at no extra cost, turning loose text ("Central, D4 to E7") into a rectangle drawn exactly in the right place on the drawing.
4. **Actionable output, not just a report** — every divergence generates a pair of annotated images (original + revised) and becomes a row in a formal PDF, ready to attach to an engineering process.
5. **Cost control by design** — image compression, purpose-scaled resolution, and the changed-pages pre-filter exist specifically to reduce tokens/latency before the AI call even happens.
