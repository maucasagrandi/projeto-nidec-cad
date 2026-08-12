"""
Script de batch para comparação de CAD Review.

Para cada pasta numerada (41–50) em:
    CAD Comparison Analysis V1.0/Sample/<n>/

Identifica os dois PDFs presentes (draw_1 e draw_2), executa o pipeline
completo de comparação e salva os três artefatos de saída em:
    CAD Comparison Analysis V1.0/Results/<n>/
        diff_pagina_1.pdf
        relatorio_ia_pagina_1.pdf
        revisado_quadrantes_pagina_1.pdf

O pipeline replica fielmente o que front.py faz no modo CAD Review.
NÃO rode via Streamlit — execute direto:
    uv run python run_batch_cad_review.py
"""

from __future__ import annotations

import os
import sys
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Imports do projeto
# ---------------------------------------------------------------------------
from prompts import system_prompt
from src.modeling.llm_models import compare_cad_pages
from src.utils.helper_func import (
    compress_png_for_llm,
    compute_visual_diff,
    count_diff_regions,
    pdf_to_images_base64,
    pdf_to_pil_images,
)
from src.utils.cad_quadrant_paint import (
    encontrar_coluna,
    extract_grid,
    paint_quadrants,
    parse_markdown_table,
)
from src.utils.cost_logger import CostLogger

# ---------------------------------------------------------------------------
# Caminhos base
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "CAD Comparison Analysis V1.0" / "Sample"
RESULT_DIR = BASE_DIR / "CAD Comparison Analysis V1.0" / "Results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

cost_logger = CostLogger(str(BASE_DIR / "custos_batch_review.csv"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pdf_pair(folder: Path) -> tuple[Path, Path]:
    """
    Devolve (pdf_original, pdf_revisado) dentro de `folder`.

    Critério: ordena os PDFs encontrados. O arquivo cujo nome contém
    "draw_1" (ou qualquer variante) é o original; "draw_2" é o revisado.
    Como fallback, usa ordem alfabética (primeiro = original).
    """
    pdfs = sorted(folder.glob("*.pdf"))
    if len(pdfs) < 2:
        raise FileNotFoundError(
            f"Esperava 2 PDFs em {folder}, encontrou {len(pdfs)}: {pdfs}"
        )
    if len(pdfs) > 2:
        print(f"  [!] {folder.name}: {len(pdfs)} PDFs encontrados — usando os dois primeiros em ordem alfabética.")

    # Tenta separar por sufixo draw_1 / draw_2
    draw1 = [p for p in pdfs if "draw_1" in p.name.lower()]
    draw2 = [p for p in pdfs if "draw_2" in p.name.lower()]

    if draw1 and draw2:
        return draw1[0], draw2[0]

    # Fallback: ordem alfabética
    return pdfs[0], pdfs[1]


def _build_pdf_report(
    report_text: str,
    page_num: int,
    item_grid,
    itens_localizacao: list,
    p1_pil_150,
    p2_pil_150,
    page_idx: int,
) -> bytes:
    """
    Replica a geração de relatório PDF do front.py.
    Retorna bytes do PDF gerado.
    """
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"Divergence Report - Page {page_num}",
        author="CAD Review Batch",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Heading1"], fontSize=14, spaceAfter=12)
    body_style  = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=6)
    cell_style  = ParagraphStyle("C", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=2)
    hdr_style   = ParagraphStyle("H", parent=styles["Normal"], fontSize=8, leading=10, fontName="Helvetica-Bold")

    story = []
    story.append(Paragraph(f"Divergence Report — Page {page_num}", title_style))
    story.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------
    # Separar linhas de texto das linhas de tabela
    # ------------------------------------------------------------------
    report_clean = report_text.replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
    lines = report_clean.split("\n")
    table_lines: list[str] = []
    text_lines: list[tuple[str, bool]] = []
    in_table = False

    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            if all(c in "-|: " for c in s):
                in_table = True
                continue
            in_table = True
            table_lines.append(s)
        else:
            if in_table and not s:
                in_table = False
            if not in_table:
                text_lines.append((s, len(table_lines) > 0))

    for line, after_table in text_lines:
        if after_table:
            break
        if line:
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "")
            story.append(Paragraph(safe, body_style))
        else:
            story.append(Spacer(1, 0.2 * cm))

    # ------------------------------------------------------------------
    # Tabela principal
    # ------------------------------------------------------------------
    parsed_rows: list[list[str]] = []
    if table_lines:
        for tl in table_lines:
            cells = [c.strip() for c in tl.split("|")[1:-1]]
            parsed_rows.append(cells)

    if parsed_rows:
        n_cols = len(parsed_rows[0])

        status_col_idx = next(
            (i for i, h in enumerate(parsed_rows[0]) if any(p in h.lower() for p in ("status", "ia"))),
            None,
        )

        table_data = []
        status_cell_styles: list[tuple[int, int, object]] = []

        for row_idx, row in enumerate(parsed_rows):
            pdf_row = []
            for col_idx, cell in enumerate(row):
                safe_cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "")
                if row_idx == 0:
                    pdf_row.append(Paragraph(safe_cell, hdr_style))
                else:
                    if ";" in safe_cell:
                        partes = [p.strip() for p in safe_cell.split(";") if p.strip()]
                        safe_cell = "<br/>".join(f"• {p}" for p in partes)
                    if status_col_idx is not None and col_idx == status_col_idx:
                        val = cell.strip().lower()
                        if "observa" in val or "observation" in val:
                            safe_cell = '<font color="#7D5A00"><b>⚠ Approved with Observation</b></font>'
                            status_cell_styles.append((row_idx, col_idx, colors.HexColor("#FEF3CD")))
                        elif "requer" in val or "correc" in val or "fixing" in val or "require" in val:
                            safe_cell = '<font color="#922B21"><b>✗ Requires Correction</b></font>'
                            status_cell_styles.append((row_idx, col_idx, colors.HexColor("#FADBD8")))
                        elif "aprovado" in val or "approved" in val:
                            safe_cell = '<font color="#1E8449"><b>✓ Approved</b></font>'
                            status_cell_styles.append((row_idx, col_idx, colors.HexColor("#D5F5E3")))
                    pdf_row.append(Paragraph(safe_cell, cell_style))
            while len(pdf_row) < n_cols:
                pdf_row.append(Paragraph("", cell_style))
            table_data.append(pdf_row)

        avail_w = landscape(A4)[0] - 3 * cm
        if n_cols == 5:
            col_widths = [avail_w * 0.05, avail_w * 0.38, avail_w * 0.18, avail_w * 0.12, avail_w * 0.27]
        elif n_cols == 4:
            col_widths = [avail_w * 0.05, avail_w * 0.42, avail_w * 0.22, avail_w * 0.31]
        else:
            col_widths = [avail_w / n_cols] * n_cols

        base_style = [
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#27AE60")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8),
            ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
            ("TOPPADDING",    (0, 0), (-1, 0),  8),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("TOPPADDING",    (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("ALIGN",         (0, 0), (0,  -1), "CENTER"),
            ("ALIGN",         (0, 0), (-1,  0), "CENTER"),
        ]
        for r_idx, c_idx, bg in status_cell_styles:
            base_style.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), bg))

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle(base_style))
        story.append(Spacer(1, 0.3 * cm))
        story.append(tbl)

    # ------------------------------------------------------------------
    # Blocos por ID (imagens anotadas original vs revisado)
    # ------------------------------------------------------------------
    tem_por_id = (
        item_grid is not None
        and itens_localizacao
        and p1_pil_150 is not None
        and p2_pil_150 is not None
        and page_idx < len(p1_pil_150)
        and page_idx < len(p2_pil_150)
        and parsed_rows
        and len(parsed_rows) > 1
    )

    if tem_por_id:
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import HRFlowable, KeepTogether

        from src.utils.cad_quadrant_paint import paint_single_item

        id_title_style = ParagraphStyle(
            "IDT", parent=styles["Heading2"], fontSize=11,
            spaceAfter=4, spaceBefore=14, textColor=colors.HexColor("#1A5276"),
        )
        id_desc_style = ParagraphStyle(
            "IDD", parent=styles["Normal"], fontSize=8,
            leading=11, spaceAfter=4, textColor=colors.HexColor("#2C3E50"),
        )
        caption_style = ParagraphStyle(
            "Cap", parent=styles["Normal"], fontSize=7,
            leading=9, textColor=colors.grey, alignment=1,
        )

        cabecalho_row = parsed_rows[0]
        col_item_idx = next(
            (i for i, h in enumerate(cabecalho_row) if any(p in h.lower() for p in ("item", "id"))), 0
        )
        col_loc_idx = next(
            (i for i, h in enumerate(cabecalho_row) if any(p in h.lower() for p in ("location", "quadrant", "localiz", "quadrante"))), None
        )
        col_dif_idx = next(
            (i for i, h in enumerate(cabecalho_row) if any(p in h.lower() for p in ("diferen", "difference", "found"))), 1
        )
        col_status_idx = next(
            (i for i, h in enumerate(cabecalho_row) if "status" in h.lower()), None
        )

        avail_w  = landscape(A4)[0] - 3 * cm
        img_w_rl = avail_w / 2.0 - 0.3 * cm
        img_h_rl = img_w_rl * (p1_pil_150[page_idx].height / p1_pil_150[page_idx].width)

        story.append(Spacer(1, 0.8 * cm))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#27AE60")))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Details by ID", title_style))

        for data_row in parsed_rows[1:]:
            id_val     = data_row[col_item_idx]  if col_item_idx  < len(data_row) else "?"
            dif_val    = data_row[col_dif_idx]   if col_dif_idx   < len(data_row) else ""
            loc_val    = data_row[col_loc_idx]   if col_loc_idx   is not None and col_loc_idx  < len(data_row) else ""
            status_val = data_row[col_status_idx] if col_status_idx is not None and col_status_idx < len(data_row) else ""

            # All flowables for this ID are collected here and wrapped in
            # KeepTogether below, so ReportLab moves the whole block to the
            # next page instead of splitting an ID's text/images across two
            # pages.
            id_block = []

            id_block.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7")))
            id_block.append(Spacer(1, 0.2 * cm))

            def _safe(t: str) -> str:
                return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "")

            safe_dif = _safe(dif_val)
            if ";" in safe_dif:
                partes   = [p.strip() for p in safe_dif.split(";") if p.strip()]
                safe_dif = "<br/>".join(f"• {p}" for p in partes)

            id_block.append(Paragraph(f"<b>ID {_safe(id_val)}</b>", id_title_style))
            id_block.append(Paragraph(safe_dif, id_desc_style))
            if loc_val:
                id_block.append(Paragraph(
                    f"<font color='#7F8C8D'>Location: {_safe(loc_val)}</font>", id_desc_style
                ))
            id_block.append(Spacer(1, 0.25 * cm))

            try:
                img1_an = paint_single_item(p1_pil_150[page_idx], id_val, loc_val, item_grid, dpi=150, status=status_val)
                img2_an = paint_single_item(p2_pil_150[page_idx], id_val, loc_val, item_grid, dpi=150, status=status_val)

                buf1 = BytesIO(); img1_an.save(buf1, format="PNG"); buf1.seek(0)
                buf2 = BytesIO(); img2_an.save(buf2, format="PNG"); buf2.seek(0)

                from reportlab.platypus import Image as RLImage
                img_rl1 = RLImage(buf1, width=img_w_rl, height=img_h_rl)
                img_rl2 = RLImage(buf2, width=img_w_rl, height=img_h_rl)

                img_tbl = Table(
                    [[img_rl1, img_rl2]],
                    colWidths=[img_w_rl + 0.3 * cm, img_w_rl + 0.3 * cm],
                )
                img_tbl.setStyle(TableStyle([
                    ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                cap_tbl = Table(
                    [[Paragraph("Original", caption_style), Paragraph("Revised", caption_style)]],
                    colWidths=[img_w_rl + 0.3 * cm, img_w_rl + 0.3 * cm],
                )
                cap_tbl.setStyle(TableStyle([
                    ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING",  (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                id_block.append(img_tbl)
                id_block.append(cap_tbl)
            except Exception as exc:
                id_block.append(Paragraph(f"<i>Images not available: {exc}</i>", id_desc_style))

            id_block.append(Spacer(1, 0.3 * cm))

            story.append(KeepTogether(id_block))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Processamento de um único folder
# ---------------------------------------------------------------------------

def process_folder(folder: Path) -> None:
    label = folder.name
    print(f"\n{'='*60}")
    print(f"  Folder: {label}")
    print(f"{'='*60}")

    # Localiza os dois PDFs
    try:
        pdf1_path, pdf2_path = _find_pdf_pair(folder)
    except FileNotFoundError as exc:
        print(f"  [ERRO] {exc}")
        return

    print(f"  Original : {pdf1_path.name}")
    print(f"  Revisado : {pdf2_path.name}")

    out_dir = RESULT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # ------------------------------------------------------------------
    # 1. Ler bytes
    # ------------------------------------------------------------------
    pdf1_bytes = pdf1_path.read_bytes()
    pdf2_bytes = pdf2_path.read_bytes()

    # ------------------------------------------------------------------
    # 2. Rasterizar
    # ------------------------------------------------------------------
    print("  [1/6] Rasterizando PDFs...")
    pages1_pil_300 = pdf_to_pil_images(pdf1_bytes, dpi=300)
    pages2_pil_300 = pdf_to_pil_images(pdf2_bytes, dpi=300)
    pages1_pil_150 = pdf_to_pil_images(pdf1_bytes, dpi=150)
    pages2_pil_150 = pdf_to_pil_images(pdf2_bytes, dpi=150)

    pages1_b64 = pdf_to_images_base64(pdf1_bytes, dpi=200)
    pages2_b64 = pdf_to_images_base64(pdf2_bytes, dpi=200)

    # ------------------------------------------------------------------
    # 3. Comprimir para LLM
    # ------------------------------------------------------------------
    print("  [2/6] Comprimindo imagens para LLM...")
    pages1_b64 = [compress_png_for_llm(img) for img in pages1_pil_300]
    pages2_b64 = [compress_png_for_llm(img) for img in pages2_pil_300]

    n_pages = min(len(pages1_b64), len(pages2_b64))

    # ------------------------------------------------------------------
    # 4. Detectar páginas com diferenças
    # ------------------------------------------------------------------
    print("  [3/6] Detectando páginas com diferenças...")
    changed_pages = []
    for i in tqdm(range(n_pages), desc="    Detectando", unit="página", leave=False):
        n_regions = count_diff_regions(pages1_pil_300[i], pages2_pil_300[i])
        if n_regions > 0:
            changed_pages.append((i, n_regions))

    if not changed_pages:
        print("  [OK] Nenhuma diferença visual detectada — pulando folder.")
        return

    print(f"  => {len(changed_pages)} página(s) com diferenças: {[i+1 for i,_ in changed_pages]}")

    # ------------------------------------------------------------------
    # 5. Processar cada página diferente
    # ------------------------------------------------------------------
    for page_idx, n_regions in tqdm(changed_pages, desc="    Processando", unit="página", leave=False):
        page_num = page_idx + 1
        print(f"\n  [4/6] Página {page_num} — diff visual...")
        diff_img = compute_visual_diff(pages1_pil_300[page_idx], pages2_pil_300[page_idx])

        print(f"  [5/6] Página {page_num} — análise LLM...")
        try:
            result, metadata = compare_cad_pages(
                image1_base64=pages1_b64[page_idx],
                image2_base64=pages2_b64[page_idx],
                system_prompt=system_prompt,
                max_tokens=32768,
            )
            cost_logger.log_analysis(metadata, page_number=page_num)
        except Exception as exc:
            print(f"  [ERRO] LLM falhou na página {page_num}: {exc}")
            continue

        print(f"  [6/6] Página {page_num} — gerando artefatos...")

        # ---- Pintura de quadrantes ----------------------------------------
        painted_img   = None
        item_grid     = None
        itens_loc     = []

        try:
            item_grid = extract_grid(pdf2_bytes, page_index=page_idx)
            if item_grid is not None:
                registros  = parse_markdown_table(result)
                col_item   = encontrar_coluna(registros[0], "item")   if registros else None
                col_local  = encontrar_coluna(registros[0], "location", "quadrant", "localiza", "quadrante") if registros else None
                col_status = encontrar_coluna(registros[0], "status") if registros else None
                if col_item and col_local:
                    itens_loc   = [(r.get(col_item, ""), r.get(col_local, "")) for r in registros]
                    status_list = [r.get(col_status, "") for r in registros] if col_status else None
                    painted_img, _ = paint_quadrants(
                        pages2_pil_300[page_idx], itens_loc, item_grid, dpi=300,
                        status_list=status_list,
                    )
        except Exception as exc:
            print(f"  [!] Pintura de quadrantes falhou: {exc}")

        # ---- Salvar diff_pagina_N.pdf ----------------------------------------
        diff_path = out_dir / f"diff_pagina_{page_num}.pdf"
        diff_rgb  = diff_img.convert("RGB") if diff_img.mode == "RGBA" else diff_img
        buf_diff  = BytesIO()
        diff_rgb.save(buf_diff, format="PDF", resolution=300)
        diff_path.write_bytes(buf_diff.getvalue())
        print(f"     Salvo: {diff_path.relative_to(BASE_DIR)}")

        # ---- Salvar revisado_quadrantes_pagina_N.pdf -------------------------
        if painted_img is not None:
            painted_path = out_dir / f"revisado_quadrantes_pagina_{page_num}.pdf"
            painted_rgb  = painted_img.convert("RGB") if painted_img.mode == "RGBA" else painted_img
            buf_painted  = BytesIO()
            painted_rgb.save(buf_painted, format="PDF", resolution=300)
            painted_path.write_bytes(buf_painted.getvalue())
            print(f"     Salvo: {painted_path.relative_to(BASE_DIR)}")
        else:
            print(f"     [!] revisado_quadrantes não gerado (grade não detectada).")

        # ---- Salvar relatorio_ia_pagina_N.pdf --------------------------------
        try:
            report_bytes = _build_pdf_report(
                report_text=result,
                page_num=page_num,
                item_grid=item_grid,
                itens_localizacao=itens_loc,
                p1_pil_150=pages1_pil_150,
                p2_pil_150=pages2_pil_150,
                page_idx=page_idx,
            )
            report_path = out_dir / f"relatorio_ia_pagina_{page_num}.pdf"
            report_path.write_bytes(report_bytes)
            print(f"     Salvo: {report_path.relative_to(BASE_DIR)}")
        except Exception as exc:
            print(f"  [ERRO] Falha ao gerar relatório PDF: {exc}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n  [OK] Folder {label} concluído em {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  BATCH CAD REVIEW")
    print(f"  Sample : {SAMPLE_DIR}")
    print(f"  Results: {RESULT_DIR}")
    print("=" * 60)

    # Coleta e ordena os folders numerados
    folders = sorted(
        [f for f in SAMPLE_DIR.iterdir() if f.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else 0,
    )

    if not folders:
        print(f"[ERRO] Nenhum folder encontrado em {SAMPLE_DIR}")
        sys.exit(1)

    print(f"\nFolders encontrados: {[f.name for f in folders]}\n")

    total_start = time.time()

    for folder in tqdm(folders, desc="Folders", unit="folder"):
        process_folder(folder)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  BATCH CONCLUÍDO em {total_elapsed:.1f}s")
    print(f"  Resultados em: {RESULT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
