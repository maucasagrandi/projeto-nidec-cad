"""Streamlit interface for the unified CAD Review workflow."""

from __future__ import annotations

import hmac
import json
import os

import cv2
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_image_zoom import image_zoom

from src.cad_review.integrated_review import run_integrated_review
from src.reporting.unified_cad_report import build_unified_report
from src.utils.opencv_cad_compare import CompareConfig

load_dotenv()
st.set_page_config(page_title="CAD Review", layout="wide")


def check_login() -> bool:
    """Keep the existing environment-based application authentication."""

    def submit() -> None:
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
    st.text_input("Usuário", key="login_user")
    st.text_input("Senha", type="password", key="login_pass")
    st.button("Entrar", on_click=submit)
    if st.session_state.get("authenticated") is False:
        st.error("Usuário ou senha incorretos")
    return False


def classification_table(classification: dict) -> list[dict[str, str]]:
    labels = {
        "classificacao": "Classificação da peça",
        "justificativa_classificacao": "Evidência da classificação",
        "lista_normas": "Normas citadas",
        "justificativas_normas": "Evidências das normas",
    }
    rows = []
    for key, value in classification.items():
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value)
        rows.append({"Campo": labels.get(key, key), "Valor extraído": str(value)})
    return rows


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if not check_login():
    st.stop()

if os.path.exists("logo.png"):
    st.sidebar.image(Image.open("logo.png"), width=280)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

st.title("CAD Review integrado")
st.write(
    "Envie o desenho original e o revisado. A classificação, as normas, as cotas e o "
    "GD&T são extraídos somente do revisado; a comparação usa os dois arquivos."
)

with st.expander("Fluxo executado"):
    st.markdown(
        """
        1. **Part Classification no revisado:** classificação da peça e normas citadas.
        2. **Cotas no revisado:** extração determinística, tabela, contador e imagem anotada.
        3. **GD&T e datums no revisado:** detecção determinística e imagem anotada.
        4. **Part Comparison:** OpenCV encontra regiões candidatas e a LLM valida/descreve as mudanças.
        5. **Relatório único:** classificação, normas, cotas, GD&T/datums e comparação.
        """
    )

left, right = st.columns(2)
with left:
    original_file = st.file_uploader("PDF original", type=["pdf"], key="original_pdf")
with right:
    revised_file = st.file_uploader("PDF revisado", type=["pdf"], key="revised_pdf")

if original_file or revised_file:
    preview_left, preview_right = st.columns(2)
    from src.utils.helper_func import pdf_to_pil_images

    if original_file:
        with preview_left:
            st.caption("Original")
            image_zoom(pdf_to_pil_images(original_file.getvalue(), dpi=100)[0])
    if revised_file:
        with preview_right:
            st.caption("Revisado")
            image_zoom(pdf_to_pil_images(revised_file.getvalue(), dpi=100)[0])

if st.button(
    "Executar revisão completa",
    disabled=not (original_file and revised_file),
    use_container_width=True,
):
    try:
        with st.spinner("Executando classificação, cotas, GD&T e comparação..."):
            result = run_integrated_review(
                original_file.getvalue(),
                revised_file.getvalue(),
                original_name=original_file.name,
                revised_name=revised_file.name,
                comparison_model="gemini-2.5-flash",
                gdt_workers=1,
                opencv_config=CompareConfig(dpi=150),
            )
            report_bytes = build_unified_report(result)
        st.session_state["integrated_review_result"] = result
        st.session_state["integrated_review_report"] = report_bytes
    # Streamlit is the application boundary: surface pipeline, credential and
    # document errors in the UI instead of terminating the server process.
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)

result = st.session_state.get("integrated_review_result")
if result is not None:
    st.divider()
    st.header("1. Part Classification")
    st.table(classification_table(result.part_classification))

    st.header("2. Normas")
    cited = result.part_classification.get("lista_normas", []) or []
    evidence = result.part_classification.get("justificativas_normas", []) or []
    if cited:
        st.subheader("Normas citadas no revisado")
        for index, standard in enumerate(cited):
            suffix = f" — {evidence[index]}" if index < len(evidence) else ""
            st.markdown(f"- {standard}{suffix}")
    else:
        st.info("Nenhuma norma explícita foi extraída.")

    suggested = result.inferred_standards.get("normas_sugeridas", []) or []
    if suggested:
        st.subheader("Normas sugeridas para validação humana")
        for standard in suggested:
            st.markdown(f"- {standard}")

    st.header("3. Cotas no revisado")
    dimension_total = sum(len(page.dimensions) for page in result.dimension_pages)
    st.metric("Total de cotas detectadas", dimension_total)
    for page in result.dimension_pages:
        st.subheader(f"Página {page.page_index + 1}")
        st.metric("Cotas nesta página", len(page.dimensions))
        if page.dimensions:
            st.table([
                {
                    "ID": dimension.dimension_id,
                    "Cota": dimension.value,
                    "Quadrante": dimension.quadrant,
                    "BBox": ", ".join(f"{value:.1f}" for value in dimension.bbox),
                }
                for dimension in page.dimensions
            ])
        if page.annotated_image is not None:
            st.image(bgr_to_rgb(page.annotated_image), use_container_width=True)

    st.header("4. GD&T e datums no revisado")
    for page in result.gdt_pages:
        st.subheader(f"Página {page.page_index + 1}")
        summary = page.report.get("summary", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("GD&T", summary.get("total_detections", 0))
        col2.metric("Referências resolvidas", summary.get("resolved_datum_refs", 0))
        col3.metric("Datums definidos", summary.get("datum_definitions_found", 0))
        if page.annotated_image is not None:
            st.image(bgr_to_rgb(page.annotated_image), use_container_width=True)

    st.header("5. Part Comparison")
    if result.paper_format_changes:
        st.subheader("Mudanças de formato do desenho")
        for change in result.paper_format_changes:
            st.warning(f"Página {change['page']}: {change['description']}")
    for page in result.comparison_pages:
        st.subheader(f"Página {page.page_index + 1}")
        if page.true_changes:
            st.table([
                {
                    "ID": change.index,
                    "Difference found": change.description,
                    "Recommended Action": "Validar a alteração com o requisito técnico aplicável.",
                }
                for change in page.true_changes
            ])
        else:
            st.success("Nenhuma mudança significativa confirmada.")
        if page.image_highlighted is not None:
            st.image(bgr_to_rgb(page.image_highlighted), use_container_width=True)

    json_bytes = json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Baixar relatório PDF",
        data=st.session_state["integrated_review_report"],
        file_name="integrated_review_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    download_right.download_button(
        "Baixar resultado JSON",
        data=json_bytes,
        file_name="integrated_review.json",
        mime="application/json",
        use_container_width=True,
    )
