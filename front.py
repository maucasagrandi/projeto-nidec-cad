import os
import time
import hmac
from io import BytesIO
import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()

from prompts import system_prompt, build_format_change_context
from src.modeling.llm_models import compare_cad_pages
from src.utils.helper_func import (
    pdf_to_images_base64,
    pdf_to_pil_images,
    compute_visual_diff,
    count_diff_regions,
    pil_to_base64,
    compress_png_for_llm,
)
from src.utils.cost_logger import CostLogger
from src.utils.cad_quadrant_paint import (
    extract_grid,
    parse_markdown_table,
    encontrar_coluna,
    paint_quadrants,
    paint_single_item,
)
from src.utils.paper_format import check_all_pages_format

# Inicializa o logger de custos
cost_logger = CostLogger("custos.csv")

# ==============================================================================
# Configuração de página
# ==============================================================================
st.set_page_config(page_title="CAD Analysis Platform", layout="wide")

# ==============================================================================
# Sidebar (Minimal)
# ==============================================================================
logo = Image.open("./logo.png")
st.sidebar.image(logo, width=280)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

# ==============================================================================
# Autenticação por usuário e senha
# ==============================================================================
def check_login() -> bool:
    """Retorna True se o usuário digitou credenciais corretos."""

    def _on_submit():
        username_ok = hmac.compare_digest(
            st.session_state["login_user"], os.getenv("APP_USERNAME", "")
        )
        password_ok = hmac.compare_digest(
            st.session_state["login_pass"], os.getenv("APP_PASSWORD", "")
        )
        if username_ok and password_ok:
            st.session_state["authenticated"] = True
            del st.session_state["login_user"]
            del st.session_state["login_pass"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated", False):
        return True

    st.markdown("### 🔐 Login")
    st.text_input("Usuário", key="login_user")
    st.text_input("Senha", type="password", key="login_pass")
    st.button("Entrar", on_click=_on_submit)

    if "authenticated" in st.session_state and not st.session_state["authenticated"]:
        st.error("😕 Usuário ou senha incorretos")
    return False


if not check_login():
    st.stop()

# ==============================================================================
# Landing Page - Seleção de Operação
# ==============================================================================
if "selected_operation" not in st.session_state:
    st.session_state.selected_operation = None

if st.session_state.selected_operation is None:
    # Exibe landing page com cards de seleção
    st.markdown(
        """
        <style>
        .operation-card {
            border: 2px solid #1f77e1;
            border-radius: 12px;
            padding: 40px 30px;
            margin: 20px 0;
            background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            min-height: 240px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .operation-card:hover {
            border-color: #58a6ff;
            background: linear-gradient(135deg, #161b22 0%, #0d47a1 100%);
            box-shadow: 0 8px 16px rgba(88, 166, 255, 0.4);
            transform: translateY(-4px);
        }
        .operation-emoji {
            font-size: 56px;
            margin-bottom: 15px;
            animation: bounce 0.6s ease-in-out infinite;
        }
        .operation-title {
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 12px;
            color: #58a6ff;
        }
        .operation-desc {
            font-size: 15px;
            color: #8b949e;
            line-height: 1.8;
            margin-bottom: 20px;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("🎯 CAD Analysis Platform")
    st.markdown("### Bem-vindo! Escolha a operação que deseja realizar")
    st.divider()
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(
            """
            <div class="operation-card">
                <div class="operation-emoji">🔄</div>
                <div class="operation-title">CAD Review</div>
                <div class="operation-desc">
                    Compare dois arquivos CAD e identifique <strong>divergências visuais e técnicas</strong> com análise de IA avançada
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→  Abrir CAD Review", key="btn_cad_review", use_container_width=True):
            st.session_state.selected_operation = "cad_review"
            st.rerun()
    
    with col2:
        st.markdown(
            """
            <div class="operation-card">
                <div class="operation-emoji">🔍</div>
                <div class="operation-title">Part Classification</div>
                <div class="operation-desc">
                    Analise uma peça individual, <strong>classifique seu tipo</strong> e <strong>extraia normas</strong> aplicadas
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→  Abrir Part Classification", key="btn_part_class", use_container_width=True):
            st.session_state.selected_operation = "part_classification"
            st.rerun()
    
    st.divider()
    st.info("💡 **Dica:** Clique no logo da MadeinWeb no sidebar a qualquer momento para retornar ao menu principal")
    
    st.stop()

# ==============================================================================
# Botão "Voltar ao Menu"
# ==============================================================================
if st.sidebar.button("🏠 Voltar ao Menu Principal"):
    st.session_state.selected_operation = None
    st.rerun()

# ==============================================================================
# CAD Review Mode
# ==============================================================================
if st.session_state.selected_operation == "cad_review":
    st.title("🔄 CAD REVIEW")
    st.write("### Validação da comparação entre dois arquivos CAD (PDF)")

    with st.expander("📋 INSTRUÇÕES"):
        st.markdown(
            """
            1. Faça o upload dos dois arquivos PDF de CAD que deseja comparar.
            2. O arquivo da **esquerda** é o desenho **original/anterior**.
            3. O arquivo da **direita** é o desenho **revisado/atual**.
            4. Clique em **Processar Comparação**.
            5. O sistema irá:
               - Identificar páginas com diferenças visuais.
               - Enviar cada par de páginas divergentes para análise pelo LLM (Gemini via GCP Vertex AI).
               - Exibir um relatório técnico de divergências por página.
               - Mostrar a sobreposição visual com as regiões alteradas marcadas em vermelho.
            """
        )

    st.divider()

    # ==============================================================================
    # Upload dos PDFs
    # ==============================================================================
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.write("#### PDF Original (versão anterior)")
        pdf1 = st.file_uploader("Upload do PDF original", type=["pdf"], key="pdf1")

    with col_up2:
        st.write("#### PDF Revisado (versão atual)")
        pdf2 = st.file_uploader("Upload do PDF revisado", type=["pdf"], key="pdf2")

    # ==============================================================================
    # Preview dos PDFs carregados (primeira página)
    # ==============================================================================
    if pdf1 or pdf2:
        st.divider()
        st.write("#### Preview da primeira página")
        prev_col1, prev_col2 = st.columns(2)

        if pdf1:
            with prev_col1:
                pages = pdf_to_pil_images(pdf1.read(), dpi=100)
                pdf1.seek(0)
                st.caption(f"Original — {len(pages)} página(s)")
                image_zoom(pages[0])

        if pdf2:
            with prev_col2:
                pages = pdf_to_pil_images(pdf2.read(), dpi=100)
                pdf2.seek(0)
                st.caption(f"Revisado — {len(pages)} página(s)")
                image_zoom(pages[0])

    st.divider()

    # ==============================================================================
    # Botão de processamento
    # ==============================================================================
    if st.button("🔄 Processar Comparação", disabled=not (pdf1 and pdf2), use_container_width=True):

        start_time = time.time()

        with st.spinner("Convertendo PDFs em imagens..."):
            pdf1_bytes = pdf1.read()
            pdf2_bytes = pdf2.read()

            pages1_b64 = pdf_to_images_base64(pdf1_bytes, dpi=200)
            pages2_b64 = pdf_to_images_base64(pdf2_bytes, dpi=200)
            pages1_pil = pdf_to_pil_images(pdf1_bytes, dpi=300)
            pages2_pil = pdf_to_pil_images(pdf2_bytes, dpi=300)

        with st.spinner("Otimizando imagens para análise..."):
            pages1_b64_compressed = []
            pages2_b64_compressed = []
            
            for i, (img_pil, b64) in enumerate(zip(pages1_pil, pages1_b64)):
                compressed = compress_png_for_llm(img_pil)
                pages1_b64_compressed.append(compressed)
            
            for i, (img_pil, b64) in enumerate(zip(pages2_pil, pages2_b64)):
                compressed = compress_png_for_llm(img_pil)
                pages2_b64_compressed.append(compressed)
            
            pages1_b64 = pages1_b64_compressed
            pages2_b64 = pages2_b64_compressed

        n_pages1 = len(pages1_b64)
        n_pages2 = len(pages2_b64)
        n_pages = min(n_pages1, n_pages2)

        if n_pages1 != n_pages2:
            st.warning(
                f"Os PDFs têm números de páginas diferentes "
                f"(original: {n_pages1}, revisado: {n_pages2}). "
                f"Serão comparadas as primeiras {n_pages} páginas."
            )

        # ==================================================================
        # Check determinístico: mudança de formato de papel (ISO 216)
        # ==================================================================
        with st.spinner("Verificando formato de papel (ISO 216)..."):
            format_changes = check_all_pages_format(pdf1_bytes, pdf2_bytes)

        with st.spinner("Identificando páginas com divergências..."):
            changed_pages = []
            for i in range(n_pages):
                n_regions = count_diff_regions(pages1_pil[i], pages2_pil[i])
                if n_regions > 0:
                    changed_pages.append((i, n_regions))

        if not changed_pages:
            st.success("✅ Nenhuma diferença visual detectada entre os dois PDFs.")
            st.stop()

        analysis_results = []

        for page_idx, n_regions in changed_pages:
            page_num = page_idx + 1

            with st.spinner(f"Gerando comparação visual da página {page_num}..."):
                diff_img = compute_visual_diff(pages1_pil[page_idx], pages2_pil[page_idx])

            with st.spinner(f"Analisando divergências com IA na página {page_num}..."):
                try:
                    # Injeta contexto de mudança de formato no prompt, se houver
                    page_format_change = format_changes.get(page_idx)
                    format_ctx = build_format_change_context(page_format_change)
                    page_prompt = system_prompt.format(format_change_context=format_ctx)

                    result, metadata = compare_cad_pages(
                        image1_base64=pages1_b64[page_idx],
                        image2_base64=pages2_b64[page_idx],
                        system_prompt=page_prompt,
                        max_tokens=32768,
                    )
                    cost_logger.log_analysis(metadata, page_number=page_num)

                    # Pintura dos quadrantes reportados pela própria LLM sobre a
                    # imagem revisada. Não faz nenhuma chamada adicional ao LLM:
                    # apenas reaproveita o texto de "Localização (Quadrante)" que
                    # já vem na tabela e a grade de zoneamento vetorial do PDF.
                    painted_img = None
                    painted_regions = None
                    grid = None
                    itens_localizacao = []
                    pages1_pil_150 = None
                    pages2_pil_150 = None
                    try:
                        grid = extract_grid(pdf2_bytes, page_index=page_idx)
                        if grid is not None:
                            registros = parse_markdown_table(result)
                            col_item   = encontrar_coluna(registros[0], "item") if registros else None
                            col_local  = encontrar_coluna(registros[0], "localiza", "quadrante") if registros else None
                            col_status = encontrar_coluna(registros[0], "status") if registros else None
                            if col_item and col_local:
                                itens_localizacao = [
                                    (reg.get(col_item, ""), reg.get(col_local, ""))
                                    for reg in registros
                                ]
                                status_list = [
                                    reg.get(col_status, "") for reg in registros
                                ] if col_status else None
                                painted_img, painted_regions = paint_quadrants(
                                    pages2_pil[page_idx],
                                    itens_localizacao,
                                    grid,
                                    dpi=300,
                                    status_list=status_list,
                                )
                        # Rasteriza ambos os PDFs em 150 dpi para os blocos por ID
                        # (resolução suficiente para leitura; muito menor que 300 dpi).
                        pages1_pil_150 = pdf_to_pil_images(pdf1_bytes, dpi=150)
                        pages2_pil_150 = pdf_to_pil_images(pdf2_bytes, dpi=150)
                    except Exception:
                        painted_img, painted_regions = None, None

                    analysis_results.append({
                        "page_num": page_num,
                        "n_regions": n_regions,
                        "diff_img": diff_img,
                        "original_img": pages1_pil[page_idx],
                        "revised_img": pages2_pil[page_idx],
                        "result": result,
                        "metadata": metadata,
                        "painted_img": painted_img,
                        "painted_regions": painted_regions,
                        "grid": grid,
                        "itens_localizacao": itens_localizacao,
                        "pages1_pil_150": pages1_pil_150,
                        "pages2_pil_150": pages2_pil_150,
                        "page_idx": page_idx,
                        "format_change": page_format_change,
                    })
                except Exception as e:
                    analysis_results.append({
                        "page_num": page_num,
                        "n_regions": n_regions,
                        "diff_img": diff_img,
                        "original_img": pages1_pil[page_idx],
                        "revised_img": pages2_pil[page_idx],
                        "result": None,
                        "error": str(e),
                        "metadata": None,
                    })

        total_time = time.time() - start_time

        st.session_state["analysis_results"] = analysis_results
        st.session_state["changed_pages"] = changed_pages
        st.session_state["total_time"] = total_time
        st.session_state["format_changes"] = format_changes

    # ==============================================================================
    # Exibição dos resultados
    # ==============================================================================
    if "analysis_results" in st.session_state:
        analysis_results = st.session_state["analysis_results"]
        changed_pages = st.session_state["changed_pages"]
        total_time = st.session_state["total_time"]
        format_changes = st.session_state.get("format_changes", {})

        st.info(
            f"**{len(changed_pages)}** página(s) com diferenças detectadas: "
            + ", ".join([f"pág. {i+1}" for i, _ in changed_pages])
        )

        # ==================================================================
        # Alertas Estruturais globais (formato de papel)
        # ==================================================================
        if format_changes:
            st.markdown("### ⚠️ Alertas Estruturais")
            for pg_idx, fc in sorted(format_changes.items()):
                alert_parts = []
                if fc.format_changed:
                    alert_parts.append(
                        f"**Formato do papel alterado** na página {pg_idx + 1}: "
                        f"`{fc.original.display_name}` → `{fc.revised.display_name}` "
                        f"({fc.original.width_mm:.0f}×{fc.original.height_mm:.0f}mm → "
                        f"{fc.revised.width_mm:.0f}×{fc.revised.height_mm:.0f}mm)"
                    )
                if fc.orientation_changed and not fc.format_changed:
                    alert_parts.append(
                        f"**Orientação alterada** na página {pg_idx + 1}: "
                        f"`{fc.original.orientation}` → `{fc.revised.orientation}`"
                    )
                for part in alert_parts:
                    st.error(f"🚨 {part}  \n**Status:** Requer Correção")
            st.divider()

        for item in analysis_results:
            page_num = item["page_num"]
            n_regions = item["n_regions"]
            diff_img = item["diff_img"]

            st.divider()
            st.write(f"### 📄 Página {page_num}")
            st.caption(f"{n_regions} região(ões) com alteração visual detectada(s)")

            # Alerta estrutural por página (formato de papel)
            page_fc = item.get("format_change")
            if page_fc:
                fc_msg_parts = []
                if page_fc.format_changed:
                    fc_msg_parts.append(
                        f"Formato: `{page_fc.original.display_name}` → "
                        f"`{page_fc.revised.display_name}`"
                    )
                if page_fc.orientation_changed and not page_fc.format_changed:
                    fc_msg_parts.append(
                        f"Orientação: `{page_fc.original.orientation}` → "
                        f"`{page_fc.revised.orientation}`"
                    )
                if fc_msg_parts:
                    st.error(
                        f"🚨 **Alerta Estrutural:** {'; '.join(fc_msg_parts)} — "
                        f"**Requer Correção**"
                    )

            painted_img = item.get("painted_img")
            painted_regions = item.get("painted_regions")

            if painted_img is not None:
                vis_col1, vis_col2, vis_col3, vis_col4 = st.columns(4)
            else:
                vis_col1, vis_col2, vis_col3 = st.columns(3)

            with vis_col1:
                st.write("###### Original")
                image_zoom(item["original_img"])
            with vis_col2:
                st.write("###### Revisado")
                image_zoom(item["revised_img"])
            with vis_col3:
                st.write("###### Diferenças")
                image_zoom(diff_img)
            if painted_img is not None:
                with vis_col4:
                    st.write("###### Revisado com Quadrantes")
                    image_zoom(painted_img)

            if painted_regions is not None:
                n_resolvidos = sum(1 for r in painted_regions if r.resolvido)
                if n_resolvidos < len(painted_regions):
                    st.caption(
                        f"⚠️ {len(painted_regions) - n_resolvidos} de {len(painted_regions)} "
                        f"item(ns) não puderam ser localizados na grade (texto de "
                        f"localização sem quadrante identificável)."
                    )

            if painted_img is not None:
                dl_col1, dl_col2, dl_col3 = st.columns(3)
            else:
                dl_col1, dl_col2 = st.columns(2)
                dl_col3 = None

            with dl_col1:
                buf_img = BytesIO()
                diff_rgb = diff_img.convert("RGB") if diff_img.mode == "RGBA" else diff_img
                diff_rgb.save(buf_img, format="PDF", resolution=300)
                buf_img.seek(0)
                st.download_button(
                    label="⬇️ Download Diff (PDF)",
                    data=buf_img,
                    file_name=f"diff_pagina_{page_num}.pdf",
                    mime="application/pdf",
                    key=f"download_diff_{page_num}",
                )

            if item.get("result"):
                with dl_col2:
                    report_text = item["result"].replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                    
                    from reportlab.lib.pagesizes import A4, landscape
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.platypus import Image as RLImage
                    from reportlab.lib.units import cm
                    from reportlab.lib import colors
                    
                    buf_report = BytesIO()
                    doc = SimpleDocTemplate(buf_report, pagesize=landscape(A4),
                                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                                            topMargin=2*cm, bottomMargin=2*cm,
                                            title=f"Relatório de Divergências - Página {page_num}",
                                            author="CAD Review - Nidec")
                    styles = getSampleStyleSheet()
                    
                    title_style = ParagraphStyle(
                        'CustomTitle', parent=styles['Heading1'],
                        fontSize=14, spaceAfter=12
                    )
                    body_style = ParagraphStyle(
                        'CustomBody', parent=styles['Normal'],
                        fontSize=9, leading=12, spaceAfter=6
                    )
                    cell_style = ParagraphStyle(
                        'CellStyle', parent=styles['Normal'],
                        fontSize=8, leading=10, spaceAfter=2
                    )
                    header_cell_style = ParagraphStyle(
                        'HeaderCell', parent=styles['Normal'],
                        fontSize=8, leading=10, fontName='Helvetica-Bold'
                    )
                    
                    story = []
                    story.append(Paragraph(f"Relatório de Divergências — Página {page_num}", title_style))
                    story.append(Spacer(1, 0.5*cm))
                    
                    lines = report_text.split("\n")
                    table_lines = []
                    text_lines = []
                    in_table = False
                    
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("|") and stripped.endswith("|"):
                            if all(c in "-|: " for c in stripped):
                                in_table = True
                                continue
                            in_table = True
                            table_lines.append(stripped)
                        else:
                            if in_table and not stripped:
                                in_table = False
                            if not in_table:
                                text_lines.append((stripped, len(table_lines) > 0))
                    
                    for line, after_table in text_lines:
                        if after_table:
                            break
                        if line:
                            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            safe_line = safe_line.replace("**", "")
                            story.append(Paragraph(safe_line, body_style))
                        else:
                            story.append(Spacer(1, 0.2*cm))
                    
                    if table_lines:
                        parsed_rows = []
                        for tl in table_lines:
                            cells = [c.strip() for c in tl.split("|")[1:-1]]
                            parsed_rows.append(cells)
                        
                        if parsed_rows:
                            n_cols = len(parsed_rows[0])

                            # Detecta índice da coluna Status IA para coloração condicional
                            status_col_idx = next(
                                (i for i, h in enumerate(parsed_rows[0])
                                 if any(p in h.lower() for p in ("status", "ia", "aprovado"))),
                                None,
                            )

                            table_data = []
                            status_cell_styles = []  # lista de (row_idx, col_idx, cor)

                            for row_idx, row in enumerate(parsed_rows):
                                pdf_row = []
                                for col_idx, cell in enumerate(row):
                                    safe_cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                    safe_cell = safe_cell.replace("**", "")
                                    if row_idx == 0:
                                        pdf_row.append(Paragraph(safe_cell, header_cell_style))
                                    else:
                                        # Bullet points em células com ponto-e-vírgula
                                        if ";" in safe_cell:
                                            partes = [p.strip() for p in safe_cell.split(";") if p.strip()]
                                            safe_cell = "<br/>".join(f"• {p}" for p in partes)
                                        # Coloração do texto na coluna Status IA
                                        if status_col_idx is not None and col_idx == status_col_idx:
                                            val = cell.strip().lower()
                                            if "observa" in val:
                                                safe_cell = f'<font color="#7D5A00"><b>⚠ Aprovado com Observação</b></font>'
                                                status_cell_styles.append((row_idx, col_idx, colors.HexColor("#FEF3CD")))
                                            elif "requer" in val or "fixing" in val or "correc" in val:
                                                safe_cell = f'<font color="#922B21"><b>✗ Requer Correção</b></font>'
                                                status_cell_styles.append((row_idx, col_idx, colors.HexColor("#FADBD8")))
                                            elif "aprovado" in val:
                                                safe_cell = f'<font color="#1E8449"><b>✓ Aprovado</b></font>'
                                                status_cell_styles.append((row_idx, col_idx, colors.HexColor("#D5F5E3")))
                                        pdf_row.append(Paragraph(safe_cell, cell_style))
                                while len(pdf_row) < n_cols:
                                    pdf_row.append(Paragraph("", cell_style))
                                table_data.append(pdf_row)
                            
                            available_width = landscape(A4)[0] - 3*cm
                            if n_cols == 5:
                                col_widths = [
                                    available_width * 0.05,  # Item
                                    available_width * 0.38,  # Diferença Encontrada
                                    available_width * 0.18,  # Localização (Quadrante)
                                    available_width * 0.12,  # Status IA
                                    available_width * 0.27,  # Ação Recomendada
                                ]
                            elif n_cols == 4:
                                col_widths = [
                                    available_width * 0.05,
                                    available_width * 0.42,
                                    available_width * 0.22,
                                    available_width * 0.31,
                                ]
                            else:
                                col_widths = [available_width / n_cols] * n_cols
                            
                            table = Table(table_data, colWidths=col_widths, repeatRows=1)
                            base_style = [
                                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 8),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                                ('TOPPADDING', (0, 0), (-1, 0), 8),
                                ('FONTSIZE', (0, 1), (-1, -1), 8),
                                ('TOPPADDING', (0, 1), (-1, -1), 5),
                                ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
                                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # cabeçalho centrado
                            ]
                            # Aplica cor de fundo nas células de Status IA
                            for r_idx, c_idx, bg_color in status_cell_styles:
                                base_style.append(('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), bg_color))
                            table.setStyle(TableStyle(base_style))
                            
                            story.append(Spacer(1, 0.3*cm))
                            story.append(table)

                    # ------------------------------------------------------------------
                    # Blocos por ID: após a tabela, um bloco para cada linha com os
                    # dois CADs marcados individualmente (só aquele ID).
                    # ------------------------------------------------------------------
                    item_grid     = item.get("grid")
                    itens_loc     = item.get("itens_localizacao", [])
                    p1_150        = item.get("pages1_pil_150")
                    p2_150        = item.get("pages2_pil_150")
                    p_idx         = item.get("page_idx", 0)

                    tem_dados_por_id = (
                        item_grid is not None
                        and itens_loc
                        and p1_150 is not None
                        and p2_150 is not None
                        and p_idx < len(p1_150)
                        and p_idx < len(p2_150)
                        and parsed_rows
                        and len(parsed_rows) > 1  # pelo menos cabeçalho + 1 dado
                    )

                    if tem_dados_por_id:
                        from reportlab.platypus import HRFlowable
                        from reportlab.lib.utils import ImageReader

                        id_title_style = ParagraphStyle(
                            'IDTitle', parent=styles['Heading2'],
                            fontSize=11, spaceAfter=4, spaceBefore=14,
                            textColor=colors.HexColor('#1A5276'),
                        )
                        id_desc_style = ParagraphStyle(
                            'IDDesc', parent=styles['Normal'],
                            fontSize=8, leading=11, spaceAfter=4,
                            textColor=colors.HexColor('#2C3E50'),
                        )
                        caption_style = ParagraphStyle(
                            'Caption', parent=styles['Normal'],
                            fontSize=7, leading=9, textColor=colors.grey,
                            alignment=1,  # centrado
                        )

                        # Cabeçalho da coluna de localização (para lookup nas linhas)
                        cabecalho_row = parsed_rows[0]
                        col_item_idx  = next(
                            (i for i, h in enumerate(cabecalho_row)
                             if any(p in h.lower() for p in ("item", "id"))),
                            0,
                        )
                        col_loc_idx = next(
                            (i for i, h in enumerate(cabecalho_row)
                             if any(p in h.lower() for p in ("localiz", "quadrante"))),
                            None,
                        )
                        col_dif_idx = next(
                            (i for i, h in enumerate(cabecalho_row)
                             if any(p in h.lower() for p in ("diferen", "difference", "found"))),
                            1,
                        )
                        col_status_idx = next(
                            (i for i, h in enumerate(cabecalho_row)
                             if "status" in h.lower()),
                            None,
                        )

                        # Largura disponível para cada imagem (duas lado a lado)
                        avail_w   = landscape(A4)[0] - 3*cm
                        img_w_rl  = avail_w / 2.0 - 0.3*cm   # largura ReportLab por imagem
                        img_h_rl  = img_w_rl * (p1_150[p_idx].height / p1_150[p_idx].width)

                        story.append(Spacer(1, 0.8*cm))
                        story.append(HRFlowable(width="100%", thickness=1.5,
                                                 color=colors.HexColor('#27AE60')))
                        story.append(Spacer(1, 0.3*cm))
                        story.append(Paragraph("Detalhamento por ID", title_style))

                        for data_row in parsed_rows[1:]:
                            id_val  = data_row[col_item_idx] if col_item_idx < len(data_row) else "?"
                            dif_val = data_row[col_dif_idx]  if col_dif_idx  < len(data_row) else ""
                            loc_val = (data_row[col_loc_idx]
                                       if col_loc_idx is not None and col_loc_idx < len(data_row)
                                       else "")
                            status_val = (data_row[col_status_idx]
                                          if col_status_idx is not None and col_status_idx < len(data_row)
                                          else "")

                            story.append(HRFlowable(width="100%", thickness=0.5,
                                                     color=colors.HexColor('#BDC3C7')))
                            story.append(Spacer(1, 0.2*cm))

                            safe_id  = id_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                            safe_dif = dif_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("**","")
                            safe_loc = loc_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

                            # Bullet points na descrição
                            if ";" in safe_dif:
                                partes   = [p.strip() for p in safe_dif.split(";") if p.strip()]
                                safe_dif = "<br/>".join(f"• {p}" for p in partes)

                            story.append(Paragraph(f"<b>ID {safe_id}</b>", id_title_style))
                            story.append(Paragraph(safe_dif, id_desc_style))
                            if safe_loc:
                                story.append(Paragraph(
                                    f"<font color='#7F8C8D'>Localização: {safe_loc}</font>",
                                    id_desc_style,
                                ))
                            story.append(Spacer(1, 0.25*cm))

                            # Gera as duas imagens anotadas com só este ID
                            try:
                                img1_anotada = paint_single_item(
                                    p1_150[p_idx], id_val, loc_val, item_grid, dpi=150,
                                    status=status_val,
                                )
                                img2_anotada = paint_single_item(
                                    p2_150[p_idx], id_val, loc_val, item_grid, dpi=150,
                                    status=status_val,
                                )

                                buf1 = BytesIO()
                                img1_anotada.save(buf1, format="PNG")
                                buf1.seek(0)

                                buf2 = BytesIO()
                                img2_anotada.save(buf2, format="PNG")
                                buf2.seek(0)

                                img_rl1 = RLImage(buf1, width=img_w_rl, height=img_h_rl)
                                img_rl2 = RLImage(buf2, width=img_w_rl, height=img_h_rl)

                                img_table = Table(
                                    [[img_rl1, img_rl2]],
                                    colWidths=[img_w_rl + 0.3*cm, img_w_rl + 0.3*cm],
                                )
                                img_table.setStyle(TableStyle([
                                    ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
                                    ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
                                    ('LEFTPADDING',  (0, 0), (-1, -1), 4),
                                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                                ]))

                                cap_table = Table(
                                    [[Paragraph("Original", caption_style),
                                      Paragraph("Revisado", caption_style)]],
                                    colWidths=[img_w_rl + 0.3*cm, img_w_rl + 0.3*cm],
                                )
                                cap_table.setStyle(TableStyle([
                                    ('ALIGN',  (0, 0), (-1, -1), 'CENTER'),
                                    ('LEFTPADDING',  (0, 0), (-1, -1), 4),
                                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                                ]))

                                story.append(img_table)
                                story.append(cap_table)
                            except Exception:
                                story.append(Paragraph(
                                    "<i>Imagens não disponíveis para este ID.</i>",
                                    id_desc_style,
                                ))

                            story.append(Spacer(1, 0.3*cm))
                    
                    found_after = False
                    for line, after_table in text_lines:
                        if after_table:
                            found_after = True
                        if found_after and line:
                            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            safe_line = safe_line.replace("**", "")
                            story.append(Spacer(1, 0.2*cm))
                            story.append(Paragraph(safe_line, body_style))
                    
                    doc.build(story)
                    buf_report.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Relatório IA (PDF)",
                        data=buf_report,
                        file_name=f"relatorio_ia_pagina_{page_num}.pdf",
                        mime="application/pdf",
                        key=f"download_report_{page_num}",
                    )

            if painted_img is not None and dl_col3 is not None:
                with dl_col3:
                    buf_painted = BytesIO()
                    painted_rgb = painted_img.convert("RGB") if painted_img.mode == "RGBA" else painted_img
                    painted_rgb.save(buf_painted, format="PDF", resolution=300)
                    buf_painted.seek(0)
                    st.download_button(
                        label="⬇️ Download Revisado com Quadrantes (PDF)",
                        data=buf_painted,
                        file_name=f"revisado_quadrantes_pagina_{page_num}.pdf",
                        mime="application/pdf",
                        key=f"download_painted_{page_num}",
                    )

            st.divider()

            if item.get("result"):
                metadata = item["metadata"]
                col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
                with col_meta1:
                    st.metric("Input Tokens", metadata.prompt_tokens)
                with col_meta2:
                    st.metric("Output Tokens", metadata.completion_tokens)
                with col_meta3:
                    st.metric("Total de Tokens", metadata.total_tokens)
                with col_meta4:
                    st.metric("Latência", f"{metadata.latency_ms:.0f}ms")

                st.markdown("#### 🔍 Relatório de Divergências")
                result_clean = item["result"].replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
                st.markdown(result_clean)
            elif item.get("error"):
                st.error(f"Erro ao analisar a página {page_num}: {item['error']}")

        st.divider()
        st.write("## 📊 Sumário da Análise")

        # Conta os status em todos os resultados desta sessão
        n_aprovado = 0
        n_observacao = 0
        n_correcao = 0
        n_alertas_estruturais = len(format_changes)
        for _item in analysis_results:
            if not _item.get("result"):
                continue
            for _reg in parse_markdown_table(_item["result"]):
                _col_st = encontrar_coluna(_reg, "status")
                _val = _reg.get(_col_st, "").strip().lower() if _col_st else ""
                if "observa" in _val:
                    n_observacao += 1
                elif "requer" in _val or "correc" in _val:
                    n_correcao += 1
                elif "aprovado" in _val:
                    n_aprovado += 1

        col_sum1, col_sum2, col_sum3, col_sum4, col_sum5, col_sum6 = st.columns(6)
        with col_sum1:
            st.metric("Páginas analisadas pelo LLM", len(changed_pages))
        with col_sum2:
            st.metric("Tempo total de processamento", f"{total_time:.1f}s")
        with col_sum3:
            st.metric("✅ Aprovado", n_aprovado)
        with col_sum4:
            st.metric("⚠️ Aprovado com Observação", n_observacao)
        with col_sum5:
            st.metric("❌ Requer Correção", n_correcao)
        with col_sum6:
            st.metric("🚨 Alertas Estruturais", n_alertas_estruturais)

        cost_summary = cost_logger.get_summary()
        
        st.divider()
        st.write("## 💰 Resumo de Custos")
        
        col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
        with col_cost1:
            st.metric("Total de Tokens", cost_summary['total_tokens'])
        with col_cost2:
            st.metric("Latência Média", cost_summary['avg_latency_ms'] + "ms" if cost_summary['avg_latency_ms'] else "N/A")
        with col_cost3:
            st.metric("Custo Total", cost_summary['total_cost'])
        with col_cost4:
            st.info(f"📁 Dados salvos em:\n`{cost_summary['file_path']}`")
        
        st.divider()
        st.info("✨ **Otimizações ativas:** Imagens PNG comprimidas com máxima compressão para reduzir tokens (~30-40% economia)")
        
        st.warning(
            "Esta aplicação pode cometer erros. Sempre valide as divergências apontadas "
            "com um profissional de engenharia."
        )

elif st.session_state.selected_operation == "part_classification":
    st.switch_page("pages/classification.py")

else:
    st.write("Faça o upload dos dois arquivos PDF para iniciar a comparação.")
