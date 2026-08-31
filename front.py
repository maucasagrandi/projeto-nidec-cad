"""Streamlit front-end for the CAD Review pipeline from the script branch.

The UI intentionally reuses the visual structure of the Part Classification page,
while the processing is delegated to the same run_review.py entrypoint used by
scripts/run_batch.py.
"""

from __future__ import annotations

import hmac
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import fitz
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_image_zoom import image_zoom

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent
_RUN_REVIEW = _PROJECT_ROOT / "run_review.py"
_LOGO_PATH = _PROJECT_ROOT / "logo.png"

st.set_page_config(page_title="CAD Review", layout="wide")

# ==============================================================================
# Theme - same green/white structure used by Part Classification
# ==============================================================================
st.markdown(
    """
    <style>
    .stApp, .main, .main > div, .block-container {
        background-color: #FFFFFF !important;
    }

    .block-container {
        max-width: 100%;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: #FFFFFF !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #13A344 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] a {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] hr {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border: none !important;
        height: 1px !important;
    }

    body, p, span, label, div {
        color: #2C3E50 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #13A344 !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #F8F9FA !important;
        border: 2px dashed #13A344 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] section:hover {
        background-color: #F0F5F2 !important;
        border-color: #0F8233 !important;
    }

    [data-testid="stFileUploader"] section button,
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        background-color: #13A344 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
    }

    [data-testid="stFileUploader"] section button:hover,
    .stButton > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #0F8233 !important;
    }

    .stButton > button:disabled {
        background-color: #BDC3C7 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stAlert"] {
        background-color: #F0F5F2 !important;
        border-radius: 8px !important;
    }

    hr {
        background-color: #D5E8DC !important;
        border: none !important;
        height: 1px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar
# ==============================================================================
if _LOGO_PATH.exists():
    st.sidebar.image(Image.open(_LOGO_PATH), width=280)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

# ==============================================================================
# Authentication - same environment-based structure as Part Classification
# ==============================================================================
def check_login() -> bool:
    def _on_submit() -> None:
        username_ok = hmac.compare_digest(
            st.session_state["login_user"], os.getenv("APP_USERNAME", "")
        )
        password_ok = hmac.compare_digest(
            st.session_state["login_pass"], os.getenv("APP_PASSWORD", "")
        )
        st.session_state["authenticated"] = username_ok and password_ok
        if username_ok and password_ok:
            del st.session_state["login_user"]
            del st.session_state["login_pass"]

    if st.session_state.get("authenticated", False):
        return True

    st.markdown("### 🔐 Login")
    st.text_input("Username", key="login_user")
    st.text_input("Password", type="password", key="login_pass")
    st.button("Sign In", on_click=_on_submit)

    if st.session_state.get("authenticated") is False:
        st.error("😕 Incorrect username or password")
    return False


if not check_login():
    st.stop()

# ==============================================================================
# Helpers
# ==============================================================================
def _first_page_preview(pdf_bytes: bytes, dpi: int = 110) -> Image.Image:
    """Rasterize only the first PDF page for a lightweight preview."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if document.page_count == 0:
            raise ValueError("PDF has no pages")
        page = document.load_page(0)
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return Image.open(BytesIO(pixmap.tobytes("png"))).copy()
    finally:
        document.close()


def _run_review(original_name: str, original_bytes: bytes, revised_name: str, revised_bytes: bytes) -> bytes:
    """Run exactly the same per-pair command used by scripts/run_batch.py."""
    if not _RUN_REVIEW.is_file():
        raise FileNotFoundError(f"run_review.py not found: {_RUN_REVIEW}")

    with tempfile.TemporaryDirectory(prefix="cad_review_streamlit_") as tmp:
        tmp_dir = Path(tmp)
        original_path = tmp_dir / Path(original_name).name
        revised_path = tmp_dir / Path(revised_name).name
        output_dir = tmp_dir / "review_results"

        original_path.write_bytes(original_bytes)
        revised_path.write_bytes(revised_bytes)

        command = [
            sys.executable,
            str(_RUN_REVIEW),
            str(original_path),
            str(revised_path),
            "-o",
            str(output_dir),
            "--opencv-dpi",
            "150",
            "--gdt-workers",
            "1",
        ]

        completed = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "Unknown pipeline error").strip()
            raise RuntimeError(details)

        report_path = output_dir / "integrated_review_report.pdf"
        if not report_path.is_file():
            raise FileNotFoundError(
                "Pipeline finished without generating integrated_review_report.pdf"
            )

        return report_path.read_bytes()


# ==============================================================================
# Main page - Part Classification structure, CAD Review behavior
# ==============================================================================
st.title("🔄 CAD Review")
st.write("### Compare two CAD drawings and generate the complete review report")

with st.expander("📋 Instructions"):
    st.markdown(
        """
        1. Upload the **original / previous** drawing on the left.
        2. Upload the **revised / current** drawing on the right.
        3. Confirm the first-page previews.
        4. Click **Generate CAD Review**.
        5. When processing finishes, download the generated PDF report.
        """
    )

st.divider()

upload_left, upload_right = st.columns(2)
with upload_left:
    st.write("#### Original PDF (previous version)")
    original_file = st.file_uploader(
        "Upload original PDF",
        type=["pdf"],
        key="cad_review_original",
    )

with upload_right:
    st.write("#### Revised PDF (current version)")
    revised_file = st.file_uploader(
        "Upload revised PDF",
        type=["pdf"],
        key="cad_review_revised",
    )

if original_file or revised_file:
    st.divider()
    st.write("#### First page preview")
    preview_left, preview_right = st.columns(2)

    if original_file:
        with preview_left:
            try:
                st.caption(f"Original — {original_file.name}")
                image_zoom(_first_page_preview(original_file.getvalue()))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not render original preview: {exc}")

    if revised_file:
        with preview_right:
            try:
                st.caption(f"Revised — {revised_file.name}")
                image_zoom(_first_page_preview(revised_file.getvalue()))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not render revised preview: {exc}")

st.divider()

if st.button(
    "🔄 Generate CAD Review",
    disabled=not (original_file and revised_file),
    use_container_width=True,
):
    # A new execution invalidates any report from a previous upload pair.
    st.session_state.pop("cad_review_report_bytes", None)
    st.session_state.pop("cad_review_report_name", None)

    try:
        with st.spinner("Running CAD Review pipeline..."):
            report_bytes = _run_review(
                original_file.name,
                original_file.getvalue(),
                revised_file.name,
                revised_file.getvalue(),
            )

        revised_stem = Path(revised_file.name).stem
        st.session_state["cad_review_report_bytes"] = report_bytes
        st.session_state["cad_review_report_name"] = f"{revised_stem}_cad_review_report.pdf"
        st.success("✅ CAD Review completed successfully.")
    except Exception as exc:  # noqa: BLE001
        st.error("CAD Review could not be completed.")
        st.exception(exc)

report_bytes = st.session_state.get("cad_review_report_bytes")
if report_bytes:
    st.divider()
    st.write("### 📄 Review Report")
    st.success("The report is ready for download.")
    st.download_button(
        "⬇️ Download CAD Review Report (PDF)",
        data=report_bytes,
        file_name=st.session_state.get(
            "cad_review_report_name",
            "integrated_review_report.pdf",
        ),
        mime="application/pdf",
        use_container_width=True,
    )

st.divider()
st.warning(
    "This application may make mistakes. Always validate the generated CAD Review "
    "with an engineering professional."
)
