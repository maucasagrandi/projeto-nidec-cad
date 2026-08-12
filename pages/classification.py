import os
import time
import hmac
import shutil
import tempfile
import traceback
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from prompts import classificacao_enriquecida_prompt
from src.cad_review.folder_pipeline import process_cad_pdf
from src.utils.helper_func import pdf_to_pil_images

# ==============================================================================
# Pipeline dependencies (paths relative to the project root)
# ==============================================================================
NORMAS_PATH = Path("Normas.xlsx")
TEMPLATE_ROOT = Path("assets/gdt/templates")
ISO1101_RULES_PATH = Path("validation/gdt/configs/iso1101_2017_reference_rules.json")
REFERENCE_CATALOG_PATH = Path("validation/gdt/reference_catalog.json")

# Static marked-drawing preview used for presentation purposes.
MARKED_DRAWING_IMAGE_PATH = Path("image.png")

# ==============================================================================
# Configuração de página
# ==============================================================================
st.set_page_config(page_title="Part Classification", layout="wide")

# ==============================================================================
# Theme Customization (CSS)
# ==============================================================================
st.markdown(
    """
    <style>
    /* Application white background */
    .main, .main > div {
        background-color: #FFFFFF !important;
    }
    
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    /* Green sidebar */
    [data-testid="stSidebar"] {
        background-color: #13A344 !important;
    }
    
    /* Sidebar content */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #13A344 !important;
    }
    
    /* Sidebar text in white for contrast */
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
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: none !important;
        height: 1px !important;
    }
    
    /* Sidebar links */
    [data-testid="stSidebar"] a {
        color: #FFFFFF !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Main content area */
    .block-container {
        background-color: #FFFFFF !important;
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    /* ===== TOP HEADER (Streamlit toolbar) ===== */
    [data-testid="stHeader"],
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E5E8EB !important;
    }
    
    [data-testid="stToolbar"] {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stToolbar"] button {
        color: #13A344 !important;
        background-color: transparent !important;
    }
    
    [data-testid="stToolbar"] button:hover {
        background-color: #E8F5EC !important;
    }
    
    [data-testid="stToolbar"] svg {
        fill: #13A344 !important;
        color: #13A344 !important;
    }
    
    [data-testid="stAppDeployButton"] {
        background-color: #13A344 !important;
        color: #FFFFFF !important;
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
    
    /* ===== FILE UPLOADER (drag and drop) ===== */
    [data-testid="stFileUploader"] section {
        background-color: #F8F9FA !important;
        border: 2px dashed #13A344 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] section:hover {
        background-color: #F0F5F2 !important;
        border-color: #0F8233 !important;
    }
    
    [data-testid="stFileUploader"] section * {
        color: #2C3E50 !important;
    }
    
    [data-testid="stFileUploader"] section small {
        color: #7F8C8D !important;
    }
    
    /* Botão "Browse files" no uploader */
    [data-testid="stFileUploader"] section button {
        background-color: #13A344 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stFileUploader"] section button:hover {
        background-color: #0F8233 !important;
    }
    
    [data-testid="stFileUploader"] section button * {
        color: #FFFFFF !important;
    }
    
    /* Ícone de cloud upload */
    [data-testid="stFileUploader"] section svg {
        color: #13A344 !important;
        fill: #13A344 !important;
    }
    
    /* Botão desabilitado */
    .stButton > button:disabled {
        background-color: #BDC3C7 !important;
        color: #FFFFFF !important;
        cursor: not-allowed !important;
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

if st.sidebar.button("🏠 Back to Main Menu"):
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
    st.text_input("Username", key="login_user")
    st.text_input("Password", type="password", key="login_pass")
    st.button("Sign In", on_click=_on_submit)

    if "authenticated" in st.session_state and not st.session_state["authenticated"]:
        st.error("😕 Incorrect username or password")
    return False


if not check_login():
    st.stop()

# ==============================================================================
# Rendering helpers
# ==============================================================================
def _render_field(label: str, field: dict | None) -> None:
    field = field or {}
    value = field.get("value")
    st.write(f"### {label}")
    if value:
        st.info(f"**Value:** {value}")
        if field.get("evidence"):
            st.caption(f"**Evidence:** {field['evidence']}")
    else:
        st.warning("Not identified")


def _is_iso_standard(cited_std: dict) -> bool:
    code = str(cited_std.get("standard") or cited_std.get("standard_raw") or "").strip().upper()
    return code.startswith("ISO")


def _render_page_image(
    path: Path,
    *,
    download_label: str | None = None,
    download_key: str | None = None,
) -> None:
    """Show an annotated page image with zoom and, optionally, a download button."""
    if not path.exists():
        st.caption(f"Image not available: {path.name}")
        return
    try:
        image_zoom(Image.open(path))
    except Exception:
        st.image(str(path), use_container_width=True)
    if download_label:
        st.download_button(
            download_label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="image/png",
            key=download_key or f"download_{path.name}",
        )


def _render_classification_tab(result: dict) -> None:
    part = result.get("part_classification") or {}
    st.write("## 🏷️ Part Classification (with Evidence)")

    col1, col2 = st.columns(2)
    with col1:
        _render_field("Component", part.get("component"))
        _render_field("Material Family", part.get("material_family"))
    with col2:
        _render_field("Document Type", part.get("document_type"))
        _render_field("Compressor Series", part.get("compressor_series"))

    review_context = result.get("review_context") or {}
    if review_context:
        st.caption(
            f"Review context — compressor series: **{review_context.get('compressor_series', 'N/A')}** "
            f"(source: {review_context.get('compressor_series_source', 'unknown')})"
        )

    st.divider()
    st.write("## 📝 Standards Cited in CAD")
    cited = [c for c in (result.get("cited_standards") or []) if not _is_iso_standard(c)]
    if cited:
        for cited_std in cited:
            label = cited_std.get("standard") or cited_std.get("standard_raw") or "?"
            with st.expander(f"📌 {label}"):
                if cited_std.get("standard_raw") and cited_std.get("standard_raw") != cited_std.get("standard"):
                    st.markdown(f"**As written:** {cited_std['standard_raw']}")
                if cited_std.get("note_number") is not None:
                    st.markdown(f"**Note:** {cited_std['note_number']}")
                st.markdown(f"**Text:** {cited_std.get('source_text', '')}")
    else:
        st.info("No cited standards found in the CAD.")


def _render_marked_drawing_tab() -> None:
    """Presentation-only preview of the static marked drawing (image.png)."""
    st.write("## 🖼️ Marked Drawing")
    _render_page_image(
        MARKED_DRAWING_IMAGE_PATH,
        download_label="⬇️ Download marked image (PNG)",
        download_key="download_marked_drawing",
    )


# ==============================================================================
# Header
# ==============================================================================
st.title("🔍 PART CLASSIFICATION")
st.write("### Structured CAD part analysis: classification and GD&T/datum evaluation")

with st.expander("📋 INSTRUCTIONS"):
    st.markdown(
        """
        1. Upload the CAD PDF file you want to analyze.
        2. The system will:
           - **Classify the part** (component, material family, compressor series, document type) with evidence
           - **Detect GD&T frames** (feature control frames) in the drawing
           - **Detect datum feature definitions** in the drawing
        3. Results are organized in tabs: Classification and Marked Drawing.
        """
    )

st.divider()

# ==============================================================================
# Upload do PDF
# ==============================================================================
st.write("#### Select the CAD file for analysis")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_classification")

# ==============================================================================
# Preview do PDF carregado
# ==============================================================================
if pdf_file:
    st.divider()
    st.write("#### First page preview")
    with st.columns(1)[0]:
        pages = pdf_to_pil_images(pdf_file.read(), dpi=100)
        pdf_file.seek(0)
        st.caption(f"Document — {len(pages)} page(s)")
        image_zoom(pages[0])

st.divider()

# ==============================================================================
# Botão de processamento
# ==============================================================================
if st.button("🚀 Analyze Part", disabled=not pdf_file, use_container_width=True):

    missing_paths = [p for p in (NORMAS_PATH, TEMPLATE_ROOT, ISO1101_RULES_PATH) if not p.exists()]
    if missing_paths:
        st.error(
            "❌ Missing required pipeline file(s): "
            + ", ".join(str(p) for p in missing_paths)
            + ". These files must exist relative to the project root before running the analysis."
        )
    else:
        # Clean up the previous run's temp folder before starting a new one.
        old_work_dir = st.session_state.get("part_classification_work_dir")
        if old_work_dir:
            shutil.rmtree(old_work_dir, ignore_errors=True)

        start_time = time.time()
        try:
            with st.spinner(
                "Running part classification, standards check, GD&T detection and datum "
                "evaluation... this can take under a minute."
            ):
                pdf_bytes = pdf_file.read()
                pdf_file.seek(0)

                work_dir = Path(tempfile.mkdtemp(prefix="cad_review_"))
                tmp_pdf_path = work_dir / pdf_file.name
                tmp_pdf_path.write_bytes(pdf_bytes)

                result = process_cad_pdf(
                    tmp_pdf_path,
                    output_dir=work_dir,
                    classification_prompt=classificacao_enriquecida_prompt,
                    normas_path=NORMAS_PATH,
                    template_root=TEMPLATE_ROOT,
                    iso1101_rules_path=ISO1101_RULES_PATH,
                    reference_catalog_path=REFERENCE_CATALOG_PATH,
                )

            total_time = time.time() - start_time

            st.session_state["part_classification_results"] = {
                "result": result,
                "output_dir": str(work_dir),
                "total_time": total_time,
                "pdf_name": pdf_file.name,
            }
            st.session_state["part_classification_work_dir"] = str(work_dir)

            st.rerun()

        except FileNotFoundError as exc:
            st.error(f"❌ Missing pipeline dependency: {exc}")
        except Exception as exc:
            st.error(f"❌ Error during analysis: {exc}")
            st.error(traceback.format_exc())

# ==============================================================================
# Exibição dos Resultados
# ==============================================================================
if "part_classification_results" in st.session_state:
    state = st.session_state["part_classification_results"]
    result = state["result"]

    st.divider()

    tab_classification, tab_marked = st.tabs(["🏷️ Classification", "🖼️ Marked Drawing"])

    with tab_classification:
        _render_classification_tab(result)

    with tab_marked:
        _render_marked_drawing_tab()

    st.divider()
    st.warning(
        "This application may make mistakes. GD&T geometry items are detector candidates. "
        "Always validate findings with an engineering professional."
    )

elif not pdf_file:
    st.info("👆 Upload a PDF file to start the analysis")
