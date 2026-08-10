import os
import os
import time
import hmac
import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()

from prompts import classificacao_enriquecida_prompt
from src.modeling.llm_models import (
    classify_cad_enriched,
    extract_text_from_pdf,
)
from src.utils.helper_func import pdf_to_pil_images
from src.utils.cost_logger import CostLogger
from src.utils.standards_applicability import (
    get_applicable_standards,
    compare_standards,
)

# Inicializa o logger de custos
cost_logger = CostLogger("custos.csv")

# ==============================================================================
# Configuração de página
# ==============================================================================
st.set_page_config(page_title="Part Classification", layout="wide")

# ==============================================================================
# Customização de Tema (CSS)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Fundo branco da aplicação */
    .main, .main > div {
        background-color: #FFFFFF !important;
    }
    
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Sidebar com cor verde */
    [data-testid="stSidebar"] {
        background-color: #13A344 !important;
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #13A344 !important;
    }
    
    /* Texto da sidebar em branco para contraste */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }
    
    /* Divisor da sidebar */
    [data-testid="stSidebar"] hr {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: none !important;
        height: 1px !important;
    }
    
    /* Links na sidebar */
    [data-testid="stSidebar"] a {
        color: #FFFFFF !important;
    }
    
    /* Botões na sidebar */
    [data-testid="stSidebar"] button {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Área de conteúdo principal */
    .block-container {
        background-color: #FFFFFF !important;
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    /* Inputs na sidebar */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* ===== CONTEÚDO PRINCIPAL ===== */
    
    /* Textos gerais */
    body, p, span, label, div {
        color: #2C3E50 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #13A344 !important;
    }
    
    /* ===== INPUTS E CAMPOS DE TEXTO ===== */
    input[type="text"],
    input[type="password"],
    input[type="email"],
    input[type="number"],
    textarea,
    [data-baseweb="input"] {
        background-color: #F8F9FA !important;
        color: #2C3E50 !important;
        border: 1px solid #D5D8DC !important;
    }
    
    input::placeholder,
    textarea::placeholder {
        color: #95A5A6 !important;
    }
    
    /* ===== BOTÕES PRIMÁRIOS ===== */
    [role="button"],
    button:not([data-testid="stSidebar"] button),
    .stButton > button {
        background-color: #13A344 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    [role="button"]:hover,
    button:hover:not([data-testid="stSidebar"] button),
    .stButton > button:hover {
        background-color: #0F8233 !important;
    }
    
    /* ===== CARDS E CONTAINERS ===== */
    [data-testid="stVerticalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"],
    .streamlit-expanderHeader {
        background-color: #F8F9FA !important;
        border-radius: 8px !important;
    }
    
    /* Cards com bordas */
    [data-testid="stContainer"] {
        border-radius: 8px !important;
        border: 1px solid #E0E0E0 !important;
    }
    
    /* ===== MENSAGENS DE ALERTA/INFO ===== */
    [data-testid="stAlert"] {
        background-color: #F0F5F2 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background-color: #E8F0EB !important;
        color: #0F5C2E !important;
    }
    
    .stSuccess {
        background-color: #E8F0EB !important;
        color: #0F8233 !important;
    }
    
    .stWarning {
        background-color: #FEF3CD !important;
        color: #856404 !important;
    }
    
    .stError {
        background-color: #FADBD8 !important;
        color: #7B241C !important;
    }
    
    /* ===== TABS ===== */
    [data-testid="stTabs"] [role="tab"] {
        background-color: #F0F2F6 !important;
        color: #2C3E50 !important;
    }
    
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background-color: #13A344 !important;
        color: #FFFFFF !important;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: #F0F5F2 !important;
        color: #13A344 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #E8F0EB !important;
    }
    
    /* ===== SELECTBOX E DROPDOWNS ===== */
    [data-baseweb="select"] {
        background-color: #F8F9FA !important;
        color: #2C3E50 !important;
    }
    
    /* ===== MÉTRICS ===== */
    [data-testid="stMetricContainer"] {
        background-color: #F0F5F2 !important;
        border-radius: 8px !important;
        border: 1px solid #D5E8DC !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2C3E50 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #13A344 !important;
    }
    
    /* ===== LINKS ===== */
    a {
        color: #13A344 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #0F8233 !important;
        text-decoration: underline !important;
    }
    
    /* ===== DIVISORES ===== */
    hr {
        background-color: #D5E8DC !important;
        border: none !important;
        height: 1px !important;
    }
    
    /* ===== PROGRESS BARS ===== */
    [data-testid="stProgress"] > div > div > div {
        background-color: #13A344 !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F0F2F6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #13A344;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #0F8233;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar
# ==============================================================================
logo = Image.open("./logo.png")
st.sidebar.image(logo, width=280)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

if st.sidebar.button("🏠 Voltar ao Menu Principal"):
    st.switch_page("front.py")

# ==============================================================================
# Autenticação
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
st.title("🔍 PART CLASSIFICATION")
st.write("### Análise estruturada de peças CAD com verificação de conformidade normativa")

with st.expander("📋 INSTRUÇÕES"):
    st.markdown(
        """
        1. Faça o upload do arquivo PDF do CAD que deseja analisar.
        2. O sistema irá:
           - **Classificar a peça** (component, material, finish, series) com evidências e confiança
           - **Determinar normas aplicáveis** via correspondência com tabela Normas.xlsx
           - **Comparar normas citadas vs esperadas** (matching, missing, unexpected)
           - **Calcular conformidade percentual** baseada nas normas obrigatórias
        3. Resultados exibidos com evidências por campo, fontes de correspondência e análise de conformidade.
        """
    )

st.divider()

# ==============================================================================
# Upload do PDF
# ==============================================================================
st.write("#### Selecione o arquivo CAD para análise")
pdf_file = st.file_uploader("Upload do PDF", type=["pdf"], key="pdf_classification")

# ==============================================================================
# Preview do PDF carregado
# ==============================================================================
if pdf_file:
    st.divider()
    st.write("#### Preview da primeira página")
    with st.columns(1)[0]:
        pages = pdf_to_pil_images(pdf_file.read(), dpi=100)
        pdf_file.seek(0)
        st.caption(f"Documento — {len(pages)} página(s)")
        image_zoom(pages[0])

st.divider()

# ==============================================================================
# Botão de processamento
# ==============================================================================
if st.button("🚀 Analisar Peça", disabled=not pdf_file, use_container_width=True):

    start_time = time.time()

    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # ------------------------------------------------------------------
        # ETAPA 1: Extração de texto do PDF
        # ------------------------------------------------------------------
        status_text.text("📄 Etapa 1/3: Extraindo texto do PDF...")
        progress_bar.progress(10)

        pdf_bytes = pdf_file.read()
        pages_pil = pdf_to_pil_images(pdf_bytes, dpi=300)
        texto_notas = extract_text_from_pdf(pdf_bytes, page_index=0)

        # ------------------------------------------------------------------
        # ETAPA 2: Classificação Enriquecida (LLM com evidências)
        # ------------------------------------------------------------------
        status_text.text("🔍 Etapa 2/3: Classificando peça com evidências...")
        progress_bar.progress(35)

        classif_result, classif_metadata = classify_cad_enriched(
            texto_notas=texto_notas,
        )
        cost_logger.log_analysis(classif_metadata, page_number=1)

        # ------------------------------------------------------------------
        # ETAPA 3: Determinação de Normas Aplicáveis e Comparação
        # ------------------------------------------------------------------
        status_text.text("📊 Etapa 3/3: Verificando conformidade normativa...")
        progress_bar.progress(60)

        # Buscar normas aplicáveis
        applicability_result = get_applicable_standards(
            component=classif_result.component.value,
            material=classif_result.material.value if classif_result.material else None,
            finish=classif_result.finish.value if classif_result.finish else None,
        )

        # Comparar normas citadas vs esperadas
        comparison_result = compare_standards(
            cited_standards=classif_result.cited_standards,
            applicable_standards=applicability_result.applicable_standards,
        )

        progress_bar.progress(100)
        status_text.text("✅ Análise concluída com sucesso!")

        total_time = time.time() - start_time

        # Salva tudo no session_state
        st.session_state["classification_results"] = {
            "classif": classif_result,
            "applicability": applicability_result,
            "comparison": comparison_result,
            "metadata": {
                "classif": classif_metadata,
            },
            "image": pages_pil[0],
            "total_time": total_time,
        }

        st.rerun()

    except Exception as e:
        progress_bar.progress(0)
        status_text.empty()
        st.error(f"❌ Erro durante análise: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# ==============================================================================
# Exibição dos Resultados
# ==============================================================================
if "classification_results" in st.session_state:
    results = st.session_state["classification_results"]
    classif = results["classif"]
    applicability = results["applicability"]
    comparison = results["comparison"]

    st.divider()
    st.write("## 📊 Resultados da Análise")

    # --- Imagem + métricas lado a lado ---
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.write("#### Imagem Analisada")
        image_zoom(results["image"])

    with col_info:
        st.write("#### Informações da Análise")

        total_input = results["metadata"]["classif"].prompt_tokens
        total_output = results["metadata"]["classif"].completion_tokens

        c1, c2, c3 = st.columns(3)
        c1.metric("Input Tokens", total_input)
        c2.metric("Output Tokens", total_output)
        c3.metric("Total Tokens", total_input + total_output)
        st.metric("Tempo Total", f"{results['total_time']:.2f}s")

    st.divider()

    # ==========================================================================
    # BLOCO 1: Classificação da Peça com Evidências
    # ==========================================================================
    st.write("## 🏷️ Classificação da Peça (com Evidências)")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Component")
        st.info(f"**Valor:** {classif.component.value}")
        st.caption(f"**Evidência:** {classif.component.evidence}")
        st.caption(f"**Confiança:** {classif.component.confidence}")

        st.write("### Material")
        if classif.material:
            st.info(f"**Valor:** {classif.material.value}")
            st.caption(f"**Evidência:** {classif.material.evidence}")
            st.caption(f"**Confiança:** {classif.material.confidence}")
        else:
            st.warning("Não identificado")

    with col2:
        st.write("### Finish")
        if classif.finish:
            st.info(f"**Valor:** {classif.finish.value}")
            st.caption(f"**Evidência:** {classif.finish.evidence}")
            st.caption(f"**Confiança:** {classif.finish.confidence}")
        else:
            st.warning("Não identificado")

        st.write("### Series")
        if classif.series:
            st.info(f"**Valor:** {classif.series.value}")
            st.caption(f"**Evidência:** {classif.series.evidence}")
            st.caption(f"**Confiança:** {classif.series.confidence}")
        else:
            st.warning("Não identificado (esperado)")

    st.divider()

    # ==========================================================================
    # BLOCO 2: Normas Citadas no CAD
    # ==========================================================================
    st.write("## 📝 Normas Citadas no CAD")

    if classif.cited_standards:
        for cited in classif.cited_standards:
            with st.expander(f"📌 {cited.standard_code}"):
                st.markdown(f"**Nota:** {cited.note_number}")
                st.markdown(f"**Texto:** {cited.note_text}")
    else:
        st.info("Nenhuma norma citada encontrada no CAD.")

    st.divider()

    # ==========================================================================
    # BLOCO 3: Normas Aplicáveis (da Tabela)
    # ==========================================================================
    st.write("## 📋 Normas Aplicáveis (Esperadas)")

    st.write(f"**Status:** {applicability.applicability_status}")
    if applicability.unresolved_fields:
        st.warning(f"⚠️ Campos não resolvidos: {', '.join(applicability.unresolved_fields)}")

    if applicability.applicable_standards:
        for app_std in applicability.applicable_standards:
            with st.expander(f"✅ {app_std.standard} — {app_std.content}"):
                st.markdown(f"**Categoria:** {app_std.category}")
                st.markdown(f"**Origem:** {app_std.source}")
                st.markdown(f"**Razão:** {app_std.reason}")
                if app_std.component_match:
                    st.caption(f"Match de componente: {app_std.component_match}")
                if app_std.material_match:
                    st.caption(f"Match de material: {app_std.material_match}")
                if app_std.finish_match:
                    st.caption(f"Match de acabamento: {app_std.finish_match}")
    else:
        st.info("Nenhuma norma aplicável encontrada (campos insuficientes ou não correspondidos).")

    st.divider()

    # ==========================================================================
    # BLOCO 4: Comparação de Normas
    # ==========================================================================
    st.write("## 🔍 Comparação: Citadas vs Esperadas")

    col_match, col_miss, col_unexp = st.columns(3)

    with col_match:
        st.metric(
            "✅ Matching",
            len(comparison.matching_standards),
        )
        if comparison.matching_standards:
            for std in comparison.matching_standards:
                st.markdown(f"- `{std}`")

    with col_miss:
        st.metric(
            "❌ Missing",
            len(comparison.missing_standards),
            delta=f"-{len(comparison.missing_standards)}" if comparison.missing_standards else None,
            delta_color="inverse",
        )
        if comparison.missing_standards:
            for std in comparison.missing_standards:
                # Buscar detalhes na lista de normas aplicáveis
                detail = next(
                    (s for s in applicability.applicable_standards if s.standard == std),
                    None
                )
                label = f"`{std}`"
                if detail:
                    label += f" — {detail.content}"
                st.markdown(f"- {label}")

    with col_unexp:
        st.metric(
            "🔸 Unexpected",
            len(comparison.unexpected_standards),
        )
        if comparison.unexpected_standards:
            for std in comparison.unexpected_standards:
                st.markdown(f"- `{std}`")

    # Barra de conformidade
    if applicability.applicable_standards:
        st.divider()
        conformity_pct = comparison.conformity_percentage
        st.write(f"### Conformidade: {conformity_pct:.1%}")
        st.progress(conformity_pct)
        
        if conformity_pct >= 0.8:
            st.success("✅ Conformidade satisfatória")
        elif conformity_pct >= 0.5:
            st.warning("⚠️ Conformidade parcial — revisar normas faltantes")
        else:
            st.error("❌ Conformidade insuficiente — múltiplas normas ausentes")

    st.divider()

    # ==========================================================================
    # BLOCO 5: JSON completo para integração
    # ==========================================================================
    st.write("## 🗂️ JSON Completo")

    resultado_json = {
        "classification": {
            "component": {
                "value": classif.component.value,
                "evidence": classif.component.evidence,
                "confidence": classif.component.confidence,
            },
            "material": {
                "value": classif.material.value if classif.material else None,
                "evidence": classif.material.evidence if classif.material else None,
                "confidence": classif.material.confidence if classif.material else None,
            },
            "finish": {
                "value": classif.finish.value if classif.finish else None,
                "evidence": classif.finish.evidence if classif.finish else None,
                "confidence": classif.finish.confidence if classif.finish else None,
            },
            "series": {
                "value": classif.series.value if classif.series else None,
                "evidence": classif.series.evidence if classif.series else None,
                "confidence": classif.series.confidence if classif.series else None,
            },
        },
        "cited_standards": [
            {
                "standard_code": cs.standard_code,
                "note_number": cs.note_number,
                "note_text": cs.note_text,
            }
            for cs in classif.cited_standards
        ],
        "applicable_standards": [
            {
                "standard": app.standard,
                "content": app.content,
                "category": app.category,
                "source": app.source,
                "reason": app.reason,
            }
            for app in applicability.applicable_standards
        ],
        "comparison": {
            "matching": comparison.matching_standards,
            "missing": comparison.missing_standards,
            "unexpected": comparison.unexpected_standards,
            "conformity_percentage": round(comparison.conformity_percentage, 3),
        },
        "metadata": {
            "applicability_status": applicability.applicability_status,
            "unresolved_fields": applicability.unresolved_fields,
            "tempo_total_segundos": round(results["total_time"], 2),
            "tokens": {
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
        },
    }

    st.json(resultado_json)

elif not pdf_file:
    st.info("👆 Faça o upload de um arquivo PDF para iniciar a análise")

