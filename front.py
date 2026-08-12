import os
import time
import hmac
from io import BytesIO
import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom
from dotenv import load_dotenv

# Load environment variables from .env file
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
from src.utils.cad_quadrant_paint import (
    extract_grid,
    parse_markdown_table,
    encontrar_coluna,
    paint_quadrants,
    paint_single_item,
)

# Initialize the cost logger
cost_logger = CostLogger("custos.csv")

# ==============================================================================
# Page configuration
# ==============================================================================
st.set_page_config(page_title="CAD Analysis Platform", layout="wide")

# ==============================================================================
# Theme Customization (CSS) - Full Green and White Theme
# ==============================================================================
st.markdown(
    """
    <style>
    /* ===== GENERAL BACKGROUND ===== */
    .stApp, .main, [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
    }
    
    .block-container {
        background-color: #FFFFFF !important;
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
    
    /* Top toolbar buttons (Deploy, menu) */
    [data-testid="stToolbar"] button {
        color: #13A344 !important;
        background-color: transparent !important;
    }
    
    [data-testid="stToolbar"] button:hover {
        background-color: #E8F5EC !important;
    }
    
    /* Toolbar icons */
    [data-testid="stToolbar"] svg {
        fill: #13A344 !important;
        color: #13A344 !important;
    }
    
    /* Deploy button */
    [data-testid="stAppDeployButton"] {
        background-color: #13A344 !important;
        color: #FFFFFF !important;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #13A344 !important;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #13A344 !important;
    }
    
    /* All sidebar text in white */
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
    }
    
    /* Sidebar links */
    [data-testid="stSidebar"] a {
        color: #FFFFFF !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.35) !important;
    }
    
    /* Sidebar inputs */
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
    
    /* Dividers */
    [data-testid="stSidebar"] hr {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: none !important;
        height: 1px !important;
    }
    
    /* ===== MAIN CONTENT ===== */
    
    /* General text */
    body, p, span, label, div {
        color: #2C3E50 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #13A344 !important;
    }
    
    /* ===== INPUTS AND TEXT FIELDS ===== */
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
    
    /* ===== PRIMARY BUTTONS ===== */
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
    
    /* Bordered cards */
    [data-testid="stContainer"] {
        border-radius: 8px !important;
        border: 1px solid #E0E0E0 !important;
    }
    
    /* ===== CUSTOM CARDS (operation-card) ===== */
    .operation-card {
        background-color: #F0F5F2 !important;
        border: 2px solid #13A344 !important;
        color: #2C3E50 !important;
    }
    
    .operation-card:hover {
        background-color: #E8F0EB !important;
        border-color: #0F8233 !important;
        box-shadow: 0 8px 16px rgba(19, 163, 68, 0.2) !important;
    }
    
    .operation-title {
        color: #13A344 !important;
    }
    
    .operation-desc {
        color: #34495E !important;
    }
    
    /* ===== ALERT/INFO MESSAGES ===== */
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
    
    /* ===== SELECTBOX AND DROPDOWNS ===== */
    [data-baseweb="select"] {
        background-color: #F8F9FA !important;
        color: #2C3E50 !important;
    }
    
    /* ===== METRICS ===== */
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
    
    /* ===== DIVIDERS ===== */
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
    
    /* "Browse files" button in the uploader */
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
    
    /* Cloud upload icon */
    [data-testid="stFileUploader"] section svg {
        color: #13A344 !important;
        fill: #13A344 !important;
    }
    
    /* ===== LARGE PRIMARY BUTTON (use_container_width=True) ===== */
    .stButton > button[kind="primary"],
    .stButton > button {
        transition: all 0.2s ease !important;
    }
    
    /* Disabled button */
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
# Sidebar (Minimal)
# ==============================================================================
logo = Image.open("./logo.png")
st.sidebar.image(logo, width=280)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

# ==============================================================================
# Username and password authentication
# ==============================================================================
def check_login() -> bool:
    """Returns True if the user entered valid credentials."""

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
# Landing Page - Operation Selection
# ==============================================================================
if "selected_operation" not in st.session_state:
    st.session_state.selected_operation = None

if st.session_state.selected_operation is None:
    # Display landing page with selection cards
    st.markdown(
        """
        <style>
        .operation-card {
            border: 2px solid #13A344;
            border-radius: 12px;
            padding: 40px 30px;
            margin: 20px 0;
            background: linear-gradient(135deg, #FFFFFF 0%, #F5F9F6 100%);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            min-height: 240px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(19, 163, 68, 0.08);
        }
        .operation-card:hover {
            border-color: #0F8233;
            background: linear-gradient(135deg, #F5F9F6 0%, #E8F5EC 100%);
            box-shadow: 0 8px 20px rgba(19, 163, 68, 0.25);
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
            color: #13A344 !important;
        }
        .operation-desc {
            font-size: 15px;
            color: #5A6C7D !important;
            line-height: 1.8;
            margin-bottom: 20px;
        }
        .operation-desc strong {
            color: #2C3E50 !important;
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
    st.markdown("### Welcome! Choose the operation you want to perform")
    st.divider()
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(
            """
            <div class="operation-card">
                <div class="operation-emoji">🔄</div>
                <div class="operation-title">CAD Review</div>
                <div class="operation-desc">
                    Compare two CAD files and identify <strong>visual and technical divergences</strong> with advanced AI analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→  Open CAD Review", key="btn_cad_review", use_container_width=True):
            st.session_state.selected_operation = "cad_review"
            st.rerun()
    
    with col2:
        st.markdown(
            """
            <div class="operation-card">
                <div class="operation-emoji">🔍</div>
                <div class="operation-title">Part Classification</div>
                <div class="operation-desc">
                    Analyze an individual part, <strong>classify its type</strong> and <strong>extract applicable standards</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("→  Open Part Classification", key="btn_part_class", use_container_width=True):
            st.session_state.selected_operation = "part_classification"
            st.rerun()
    
    st.divider()
    st.info("💡 **Tip:** Click the MadeinWeb logo in the sidebar at any time to return to the main menu")
    
    st.stop()

# ==============================================================================
# "Back to Menu" Button
# ==============================================================================
if st.sidebar.button("🏠 Back to Main Menu"):
    st.session_state.selected_operation = None
    st.rerun()

# ==============================================================================
# CAD Review Mode
# ==============================================================================
if st.session_state.selected_operation == "cad_review":
    st.title("🔄 CAD REVIEW")
    st.write("### Validation of comparison between two CAD files (PDF)")

    with st.expander("📋 INSTRUCTIONS"):
        st.markdown(
            """
            1. Upload the two CAD PDF files you want to compare.
            2. The file on the **left** is the **original/previous** drawing.
            3. The file on the **right** is the **revised/current** drawing.
            4. Click on **Process Comparison**.
            5. The system will:
               - Identify pages with visual differences.
               - Send each pair of divergent pages for analysis by the LLM (Gemini via GCP Vertex AI).
               - Display a technical divergence report for each page.
               - Show the visual overlay with altered regions marked in red.
            """
        )

    st.divider()

    # ==============================================================================
    # PDF Upload
    # ==============================================================================
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.write("#### Original PDF (previous version)")
        pdf1 = st.file_uploader("Upload original PDF", type=["pdf"], key="pdf1")

    with col_up2:
        st.write("#### Revised PDF (current version)")
        pdf2 = st.file_uploader("Upload revised PDF", type=["pdf"], key="pdf2")

    # ==============================================================================
    # Preview of loaded PDFs (first page)
    # ==============================================================================
    if pdf1 or pdf2:
        st.divider()
        st.write("#### First page preview")
        prev_col1, prev_col2 = st.columns(2)

        if pdf1:
            with prev_col1:
                pages = pdf_to_pil_images(pdf1.read(), dpi=100)
                pdf1.seek(0)
                st.caption(f"Original — {len(pages)} page(s)")
                image_zoom(pages[0])

        if pdf2:
            with prev_col2:
                pages = pdf_to_pil_images(pdf2.read(), dpi=100)
                pdf2.seek(0)
                st.caption(f"Revised — {len(pages)} page(s)")
                image_zoom(pages[0])

    st.divider()

    # ==============================================================================
    # Processing button
    # ==============================================================================
    if st.button("🔄 Process Comparison", disabled=not (pdf1 and pdf2), use_container_width=True):

        start_time = time.time()

        with st.spinner("Converting PDFs to images..."):
            pdf1_bytes = pdf1.read()
            pdf2_bytes = pdf2.read()

            pages1_b64 = pdf_to_images_base64(pdf1_bytes, dpi=200)
            pages2_b64 = pdf_to_images_base64(pdf2_bytes, dpi=200)
            pages1_pil = pdf_to_pil_images(pdf1_bytes, dpi=300)
            pages2_pil = pdf_to_pil_images(pdf2_bytes, dpi=300)

        with st.spinner("Optimizing images for analysis..."):
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
                f"The PDFs have different page counts "
                f"(original: {n_pages1}, revised: {n_pages2}). "
                f"The first {n_pages} pages will be compared."
            )

        with st.spinner("Identifying pages with divergences..."):
            changed_pages = []
            for i in range(n_pages):
                n_regions = count_diff_regions(pages1_pil[i], pages2_pil[i])
                if n_regions > 0:
                    changed_pages.append((i, n_regions))

        if not changed_pages:
            st.success("✅ No visual differences detected between the two PDFs.")
            st.stop()

        analysis_results = []

        for page_idx, n_regions in changed_pages:
            page_num = page_idx + 1

            with st.spinner(f"Generating visual comparison of page {page_num}..."):
                diff_img = compute_visual_diff(pages1_pil[page_idx], pages2_pil[page_idx])

            with st.spinner(f"Analyzing divergences with AI on page {page_num}..."):
                try:
                    result, metadata = compare_cad_pages(
                        image1_base64=pages1_b64[page_idx],
                        image2_base64=pages2_b64[page_idx],
                        system_prompt=system_prompt,
                        max_tokens=32768,
                    )
                    cost_logger.log_analysis(metadata, page_number=page_num)

                    # Paint the quadrants reported by the LLM itself on the
                    # revised image. No additional LLM call is made: it just
                    # reuses the "Location (Quadrant)" text that already
                    # comes in the table and the PDF's vector zoning grid.
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
                            col_local  = encontrar_coluna(registros[0], "location", "quadrant", "localiza", "quadrante") if registros else None
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
                        # Rasterize both PDFs at 150 dpi for per-ID blocks
                        # (resolution sufficient for reading; much smaller than 300 dpi).
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

    # ==============================================================================
    # Display of results
    # ==============================================================================
    if "analysis_results" in st.session_state:
        analysis_results = st.session_state["analysis_results"]
        changed_pages = st.session_state["changed_pages"]

        st.info(
            f"**{len(changed_pages)}** page(s) with detected differences: "
            + ", ".join([f"p. {i+1}" for i, _ in changed_pages])
        )

        for item in analysis_results:
            page_num = item["page_num"]
            n_regions = item["n_regions"]

            st.divider()
            st.write(f"### 📄 Page {page_num}")
            st.caption(f"{n_regions} region(s) with detected visual changes")

            painted_img = item.get("painted_img")
            painted_regions = item.get("painted_regions")

            if painted_img is not None:
                st.write("###### Revised with Quadrants")
                image_zoom(painted_img)
            else:
                st.write("###### Revised")
                image_zoom(item["revised_img"])

            if painted_regions is not None:
                n_resolvidos = sum(1 for r in painted_regions if r.resolvido)
                if n_resolvidos < len(painted_regions):
                    st.caption(
                        f"⚠️ {len(painted_regions) - n_resolvidos} of {len(painted_regions)} "
                        f"item(s) could not be located on the grid (location text "
                        f"without identifiable quadrant)."
                    )

            if item.get("result"):
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
                                        title=f"Divergence Report - Page {page_num}",
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
                story.append(Paragraph(f"Divergence Report — Page {page_num}", title_style))
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

                        # Detect Status IA column index for conditional coloring
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
                                    # Bullet points in cells with semicolons
                                    if ";" in safe_cell:
                                        partes = [p.strip() for p in safe_cell.split(";") if p.strip()]
                                        safe_cell = "<br/>".join(f"• {p}" for p in partes)
                                    # Text coloring in the Status IA column
                                    if status_col_idx is not None and col_idx == status_col_idx:
                                        val = cell.strip().lower()
                                        if "observa" in val or "observation" in val:
                                            safe_cell = f'<font color="#7D5A00"><b>⚠ Approved with Observation</b></font>'
                                            status_cell_styles.append((row_idx, col_idx, colors.HexColor("#FEF3CD")))
                                        elif "requer" in val or "fixing" in val or "correc" in val or "require" in val:
                                            safe_cell = f'<font color="#922B21"><b>✗ Requires Correction</b></font>'
                                            status_cell_styles.append((row_idx, col_idx, colors.HexColor("#FADBD8")))
                                        elif "aprovado" in val or "approved" in val:
                                            safe_cell = f'<font color="#1E8449"><b>✓ Approved</b></font>'
                                            status_cell_styles.append((row_idx, col_idx, colors.HexColor("#D5F5E3")))
                                    pdf_row.append(Paragraph(safe_cell, cell_style))
                            while len(pdf_row) < n_cols:
                                pdf_row.append(Paragraph("", cell_style))
                            table_data.append(pdf_row)
                        
                        available_width = landscape(A4)[0] - 3*cm
                        if n_cols == 5:
                            col_widths = [
                                available_width * 0.05,  # Item
                                available_width * 0.38,  # Difference Found
                                available_width * 0.18,  # Location (Quadrant)
                                available_width * 0.12,  # AI Status
                                available_width * 0.27,  # Recommended Action
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
                            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),  # centered header
                        ]
                        # Apply background color to AI Status cells
                        for r_idx, c_idx, bg_color in status_cell_styles:
                            base_style.append(('BACKGROUND', (c_idx, r_idx), (c_idx, r_idx), bg_color))
                        table.setStyle(TableStyle(base_style))
                        
                        story.append(Spacer(1, 0.3*cm))
                        story.append(table)

                # ------------------------------------------------------------------
                # Per-ID blocks: after the table, one block for each row with
                # the two CADs marked individually (only that ID).
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
                    and len(parsed_rows) > 1  # at least header + 1 data row
                )

                if tem_dados_por_id:
                    from reportlab.platypus import HRFlowable, KeepTogether
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

                    # Header of the location column (for lookup in rows)
                    cabecalho_row = parsed_rows[0]
                    col_item_idx  = next(
                        (i for i, h in enumerate(cabecalho_row)
                         if any(p in h.lower() for p in ("item", "id"))),
                        0,
                    )
                    col_loc_idx = next(
                        (i for i, h in enumerate(cabecalho_row)
                         if any(p in h.lower() for p in ("location", "quadrant", "localiz", "quadrante"))),
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

                    # Available width for each image (two side by side)
                    avail_w   = landscape(A4)[0] - 3*cm
                    img_w_rl  = avail_w / 2.0 - 0.3*cm   # ReportLab width per image
                    img_h_rl  = img_w_rl * (p1_150[p_idx].height / p1_150[p_idx].width)

                    story.append(Spacer(1, 0.8*cm))
                    story.append(HRFlowable(width="100%", thickness=1.5,
                                             color=colors.HexColor('#27AE60')))
                    story.append(Spacer(1, 0.3*cm))
                    story.append(Paragraph("Details by ID", title_style))

                    for data_row in parsed_rows[1:]:
                        id_val  = data_row[col_item_idx] if col_item_idx < len(data_row) else "?"
                        dif_val = data_row[col_dif_idx]  if col_dif_idx  < len(data_row) else ""
                        loc_val = (data_row[col_loc_idx]
                                   if col_loc_idx is not None and col_loc_idx < len(data_row)
                                   else "")
                        status_val = (data_row[col_status_idx]
                                      if col_status_idx is not None and col_status_idx < len(data_row)
                                      else "")

                        # All flowables for this ID are collected here and wrapped in
                        # KeepTogether below, so ReportLab moves the whole block to the
                        # next page instead of splitting an ID's text/images across two
                        # pages.
                        id_block = []

                        id_block.append(HRFlowable(width="100%", thickness=0.5,
                                                    color=colors.HexColor('#BDC3C7')))
                        id_block.append(Spacer(1, 0.2*cm))

                        safe_id  = id_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        safe_dif = dif_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("**","")
                        safe_loc = loc_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

                        # Bullet points in the description
                        if ";" in safe_dif:
                            partes   = [p.strip() for p in safe_dif.split(";") if p.strip()]
                            safe_dif = "<br/>".join(f"• {p}" for p in partes)

                        id_block.append(Paragraph(f"<b>ID {safe_id}</b>", id_title_style))
                        id_block.append(Paragraph(safe_dif, id_desc_style))
                        if safe_loc:
                            id_block.append(Paragraph(
                                f"<font color='#7F8C8D'>Location: {safe_loc}</font>",
                                id_desc_style,
                            ))
                        id_block.append(Spacer(1, 0.25*cm))

                        # Generate the two annotated images with only this ID
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
                                  Paragraph("Revised", caption_style)]],
                                colWidths=[img_w_rl + 0.3*cm, img_w_rl + 0.3*cm],
                            )
                            cap_table.setStyle(TableStyle([
                                ('ALIGN',  (0, 0), (-1, -1), 'CENTER'),
                                ('LEFTPADDING',  (0, 0), (-1, -1), 4),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                            ]))

                            id_block.append(img_table)
                            id_block.append(cap_table)
                        except Exception:
                            id_block.append(Paragraph(
                                "<i>Images not available for this ID.</i>",
                                id_desc_style,
                            ))

                        id_block.append(Spacer(1, 0.3*cm))

                        story.append(KeepTogether(id_block))
                
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
                    label="⬇️ Download AI Report (PDF)",
                    data=buf_report,
                    file_name=f"ai_report_page_{page_num}.pdf",
                    mime="application/pdf",
                    key=f"download_report_{page_num}",
                )

            if item.get("error"):
                st.divider()
                st.error(f"Error analyzing page {page_num}: {item['error']}")

        st.divider()
        st.warning(
            "This application may make mistakes. Always validate the identified divergences "
            "with an engineering professional."
        )

elif st.session_state.selected_operation == "part_classification":
    st.switch_page("pages/classification.py")

else:
    st.write("Upload both PDF files to start the comparison.")
