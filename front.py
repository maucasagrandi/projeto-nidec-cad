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

from prompts import system_prompt
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

# Inicializa o logger de custos
cost_logger = CostLogger("custos.csv")

# ==============================================================================
# Configuração de página
# ==============================================================================
st.set_page_config(page_title="Comparação de CAD", layout="wide")

# ==============================================================================
# Sidebar
# ==============================================================================
logo = Image.open("./logo.png")
st.sidebar.image(logo, width=300)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

# ==============================================================================
# Autenticação por usuário e senha
# ==============================================================================
def check_login() -> bool:
    """Retorna True se o usuário digitou credenciais corretas."""

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
# Header
# ==============================================================================
st.title("CAD REVIEW")
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
           - Enviar cada par de páginas divergentes para análise pelo LLM (Claude via GCP Vertex AI).
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
            pdf1.seek(0)  # resetar para uso posterior
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
if st.button("🔄 Processar Comparação", disabled=not (pdf1 and pdf2)):

    start_time = time.time()

    with st.spinner("Convertendo PDFs em imagens..."):
        pdf1_bytes = pdf1.read()
        pdf2_bytes = pdf2.read()

        # Renderiza em DPI adequado para análise: 200 para LLM, 300 para diff visual
        pages1_b64 = pdf_to_images_base64(pdf1_bytes, dpi=200)
        pages2_b64 = pdf_to_images_base64(pdf2_bytes, dpi=200)
        pages1_pil = pdf_to_pil_images(pdf1_bytes, dpi=300)
        pages2_pil = pdf_to_pil_images(pdf2_bytes, dpi=300)

    # Comprime imagens para o LLM (reduz tokens ~30-40%)
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

    # Pré-filtragem: identifica páginas com diferenças visuais
    with st.spinner("Identificando páginas com divergências..."):
        changed_pages = []
        for i in range(n_pages):
            n_regions = count_diff_regions(pages1_pil[i], pages2_pil[i])
            if n_regions > 0:
                changed_pages.append((i, n_regions))

    if not changed_pages:
        st.success("✅ Nenhuma diferença visual detectada entre os dois PDFs.")
        st.stop()

    # Análise por LLM para cada página divergente
    analysis_results = []

    for page_idx, n_regions in changed_pages:
        page_num = page_idx + 1

        # Diff visual
        with st.spinner(f"Gerando comparação visual da página {page_num}..."):
            diff_img = compute_visual_diff(pages1_pil[page_idx], pages2_pil[page_idx])

        # Análise LLM
        with st.spinner(f"Analisando divergências com IA na página {page_num}..."):
            try:
                result, metadata = compare_cad_pages(
                    image1_base64=pages1_b64[page_idx],
                    image2_base64=pages2_b64[page_idx],
                    system_prompt=system_prompt,
                    max_tokens=32768,
                )
                cost_logger.log_analysis(metadata, page_number=page_num)
                
                analysis_results.append({
                    "page_num": page_num,
                    "n_regions": n_regions,
                    "diff_img": diff_img,
                    "original_img": pages1_pil[page_idx],
                    "revised_img": pages2_pil[page_idx],
                    "result": result,
                    "metadata": metadata,
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

    # Salva tudo no session_state para persistir entre reruns
    st.session_state["analysis_results"] = analysis_results
    st.session_state["changed_pages"] = changed_pages
    st.session_state["total_time"] = total_time

# ==============================================================================
# Exibição dos resultados (persistente via session_state)
# ==============================================================================
if "analysis_results" in st.session_state:
    analysis_results = st.session_state["analysis_results"]
    changed_pages = st.session_state["changed_pages"]
    total_time = st.session_state["total_time"]

    st.info(
        f"**{len(changed_pages)}** página(s) com diferenças detectadas: "
        + ", ".join([f"pág. {i+1}" for i, _ in changed_pages])
    )

    for item in analysis_results:
        page_num = item["page_num"]
        n_regions = item["n_regions"]
        diff_img = item["diff_img"]

        st.divider()
        st.write(f"### 📄 Página {page_num}")
        st.caption(f"{n_regions} região(ões) com alteração visual detectada(s)")

        # Exibição side-by-side
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

        # Botões de download
        dl_col1, dl_col2 = st.columns(2)

        # Download da imagem diff em PDF
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

        # Download do relatório da IA em PDF
        if item.get("result"):
            with dl_col2:
                report_text = item["result"].replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
                
                # Parseia o markdown: separa texto antes/depois da tabela
                lines = report_text.split("\n")
                table_lines = []
                text_lines = []
                in_table = False
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("|") and stripped.endswith("|"):
                        # Ignora linha separadora (|---|---|...)
                        if all(c in "-| " for c in stripped):
                            in_table = True
                            continue
                        in_table = True
                        table_lines.append(stripped)
                    else:
                        if in_table and not stripped:
                            in_table = False
                        if not in_table:
                            text_lines.append((stripped, len(table_lines) > 0))
                
                # Adiciona texto que vem antes da tabela
                for line, after_table in text_lines:
                    if after_table:
                        break
                    if line:
                        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        # Remove markdown bold
                        safe_line = safe_line.replace("**", "")
                        story.append(Paragraph(safe_line, body_style))
                    else:
                        story.append(Spacer(1, 0.2*cm))
                
                # Constrói a tabela formatada
                if table_lines:
                    # Parseia células
                    parsed_rows = []
                    for tl in table_lines:
                        cells = [c.strip() for c in tl.split("|")[1:-1]]
                        parsed_rows.append(cells)
                    
                    if parsed_rows:
                        # Determina número de colunas
                        n_cols = len(parsed_rows[0])
                        
                        # Converte cada célula em Paragraph para wrapping
                        table_data = []
                        for row_idx, row in enumerate(parsed_rows):
                            pdf_row = []
                            for cell in row:
                                safe_cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                safe_cell = safe_cell.replace("**", "")
                                if row_idx == 0:
                                    pdf_row.append(Paragraph(safe_cell, header_cell_style))
                                else:
                                    pdf_row.append(Paragraph(safe_cell, cell_style))
                            # Preenche colunas faltantes se necessário
                            while len(pdf_row) < n_cols:
                                pdf_row.append(Paragraph("", cell_style))
                            table_data.append(pdf_row)
                        
                        # Define larguras proporcionais das colunas
                        available_width = landscape(A4)[0] - 3*cm
                        if n_cols == 5:
                            col_widths = [
                                available_width * 0.05,   # Item
                                available_width * 0.35,   # Diferença
                                available_width * 0.20,   # Localização
                                available_width * 0.18,   # Tipo
                                available_width * 0.22,   # Impacto
                            ]
                        else:
                            col_widths = [available_width / n_cols] * n_cols
                        
                        # Cria tabela
                        table = Table(table_data, colWidths=col_widths, repeatRows=1)
                        table.setStyle(TableStyle([
                            # Header
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 8),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                            ('TOPPADDING', (0, 0), (-1, 0), 8),
                            # Body
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('TOPPADDING', (0, 1), (-1, -1), 5),
                            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
                            ('LEFTPADDING', (0, 0), (-1, -1), 6),
                            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                            # Grid
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
                            # Alinhamento
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                        ]))
                        
                        story.append(Spacer(1, 0.3*cm))
                        story.append(table)
                
                # Texto após a tabela
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

        st.divider()

        # Análise LLM
        if item.get("result"):
            metadata = item["metadata"]
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.metric("Total de Tokens", metadata.total_tokens)
            with col_meta2:
                st.metric("Latência", f"{metadata.latency_ms:.0f}ms")
            with col_meta3:
                cost = cost_logger.calculate_cost(metadata)
                st.metric("Custo Estimado", f"${cost:.6f}")

            st.markdown("#### 🔍 Relatório de Divergências")
            result_clean = item["result"].replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
            st.markdown(result_clean)
        elif item.get("error"):
            st.error(f"Erro ao analisar a página {page_num}: {item['error']}")

    # Sumário final
    st.divider()
    st.write("## 📊 Sumário da Análise")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Páginas analisadas pelo LLM", len(changed_pages))
    with col_sum2:
        st.metric("Tempo total de processamento", f"{total_time:.1f}s")
    with col_sum3:
        cost_summary = cost_logger.get_summary()
        st.metric("Análises realizadas", cost_summary['total_analyses'])
    
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

elif not (pdf1 and pdf2):
    st.write("Faça o upload dos dois arquivos PDF para iniciar a comparação.")
