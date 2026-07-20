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
        # Converte base64 → PIL → comprime → volta base64
        pages1_b64_compressed = []
        pages2_b64_compressed = []
        
        for i, (img_pil, b64) in enumerate(zip(pages1_pil, pages1_b64)):
            compressed = compress_png_for_llm(img_pil)
            pages1_b64_compressed.append(compressed)
        
        for i, (img_pil, b64) in enumerate(zip(pages2_pil, pages2_b64)):
            compressed = compress_png_for_llm(img_pil)
            pages2_b64_compressed.append(compressed)
        
        # Usa imagens comprimidas para LLM
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

    # ------------------------------------------------------------------
    # Pré-filtragem: identifica páginas com diferenças visuais
    # ------------------------------------------------------------------
    with st.spinner("Identificando páginas com divergências..."):
        changed_pages = []
        for i in range(n_pages):
            n_regions = count_diff_regions(pages1_pil[i], pages2_pil[i])
            if n_regions > 0:
                changed_pages.append((i, n_regions))

    if not changed_pages:
        st.success("✅ Nenhuma diferença visual detectada entre os dois PDFs.")
        st.stop()

    st.info(
        f"**{len(changed_pages)}** página(s) com diferenças detectadas de {n_pages} total: "
        + ", ".join([f"pág. {i+1}" for i, _ in changed_pages])
    )

    # ------------------------------------------------------------------
    # Análise por LLM para cada página divergente
    # ------------------------------------------------------------------
    all_results = []

    for page_idx, n_regions in changed_pages:
        page_num = page_idx + 1
        st.divider()
        st.write(f"### 📄 Página {page_num}")
        st.caption(f"{n_regions} região(ões) com alteração visual detectada(s)")

        # Diff visual
        with st.spinner(f"Gerando comparação visual da página {page_num}..."):
            diff_img = compute_visual_diff(pages1_pil[page_idx], pages2_pil[page_idx])

        # Exibição side-by-side
        vis_col1, vis_col2, vis_col3 = st.columns(3)
        with vis_col1:
            st.write("###### Original")
            image_zoom(pages1_pil[page_idx])
        with vis_col2:
            st.write("###### Revisado")
            image_zoom(pages2_pil[page_idx])
        with vis_col3:
            st.write("###### Diferenças (marcadas em vermelho)")
            image_zoom(diff_img)

        # Botão de download do diff em PDF (300 DPI)
        with vis_col1:
            buf = BytesIO()
            diff_rgb = diff_img.convert("RGB") if diff_img.mode == "RGBA" else diff_img
            diff_rgb.save(buf, format="PDF", resolution=300)
            buf.seek(0)
            st.download_button(
                label="⬇️ Download Diff (PDF)",
                data=buf,
                file_name=f"diff_pagina_{page_num}.pdf",
                mime="application/pdf",
                key=f"download_diff_{page_num}",
            )

        st.divider()
        # Análise LLM
        with st.spinner(f"Analisando divergências com IA na página {page_num}..."):
            try:
                result, metadata = compare_cad_pages(
                    image1_base64=pages1_b64[page_idx],
                    image2_base64=pages2_b64[page_idx],
                    system_prompt=system_prompt,
                    max_tokens=32768,
                )
                
                # Log de custos
                cost_logger.log_analysis(metadata, page_number=page_num)
                
                # Exibe metadados
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.metric("Total de Tokens", metadata.total_tokens)
                with col_meta2:
                    st.metric("Latência", f"{metadata.latency_ms:.0f}ms")
                with col_meta3:
                    cost = cost_logger.calculate_cost(metadata)
                    st.metric("Custo Estimado", f"${cost:.6f}")
                
                # Exibe relatório de divergências
                all_results.append((page_num, result))
                
                st.markdown("#### 🔍 Relatório de Divergências")
                # Sanitiza tags HTML que o modelo pode retornar indevidamente
                result_clean = result.replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
                st.markdown(result_clean)
                
            except Exception as e:
                st.error(f"Erro ao analisar a página {page_num}: {e}")

    # ------------------------------------------------------------------
    # Sumário final
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    st.divider()
    st.write("## 📊 Sumário da Análise")
    
    # Métricas de análise
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Páginas analisadas pelo LLM", len(changed_pages))
    with col_sum2:
        st.metric("Tempo total de processamento", f"{total_time:.1f}s")
    with col_sum3:
        cost_summary = cost_logger.get_summary()
        st.metric("Análises realizadas", cost_summary['total_analyses'])
    
    # Resumo de custos
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
    
    # Info sobre otimização
    st.divider()
    st.info("✨ **Otimizações ativas:** Imagens PNG comprimidas com máxima compressão para reduzir tokens (~30-40% economia)")
    
    st.warning(
        "Esta aplicação pode cometer erros. Sempre valide as divergências apontadas "
        "com um profissional de engenharia."
    )

elif not (pdf1 and pdf2):
    st.write("Faça o upload dos dois arquivos PDF para iniciar a comparação.")
