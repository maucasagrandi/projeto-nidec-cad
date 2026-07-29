import os
import time
import hmac
import streamlit as st
from PIL import Image
from streamlit_image_zoom import image_zoom
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()

from prompts import classificacao_e_normas_prompt
from src.modeling.llm_models import (
    classify_and_extract_norms,
    infer_missing_norms,
    extract_text_from_pdf,
)
from src.utils.helper_func import (
    pdf_to_pil_images,
)
from src.utils.cost_logger import CostLogger
from src.utils.json_display import (
    display_json_card,
    display_json_expandable,
    create_json_report,
)

# Inicializa o logger de custos
cost_logger = CostLogger("custos.csv")

# ==============================================================================
# Configuração de página
# ==============================================================================
st.set_page_config(page_title="Part Classification", layout="wide")

# ==============================================================================
# Sidebar (Minimal)
# ==============================================================================
logo = Image.open("./logo.png")
st.sidebar.image(logo, width=280)
st.sidebar.divider()
st.sidebar.markdown("#### Powered by [MadeinWeb](https://madeinweb.com.br/)")

# Botão para voltar ao menu
if st.sidebar.button("🏠 Voltar ao Menu Principal"):
    st.switch_page("front.py")

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
st.title("🔍 PART CLASSIFICATION")
st.write("### Análise estruturada de peças CAD com extração de normas")

with st.expander("📋 INSTRUÇÕES"):
    st.markdown(
        """
        1. Faça o upload do arquivo PDF do CAD que deseja analisar.
        2. O sistema irá:
           - **Classificar a peça** baseado no texto extraído do CAD
           - **Extrair normas** mencionadas nas NOTES do CAD
           - **Inferir normas faltantes** baseado no tipo de peça
        3. Resultados serão exibidos em formato estruturado JSON para integração com sistemas externos.
        """
    )

st.divider()

# ==============================================================================
# Upload do PDF
# ==============================================================================
st.write("#### Selecione o arquivo CAD para análise")
pdf_file = st.file_uploader("Upload do PDF", type=["pdf"], key="pdf_classification")

# ==============================================================================
# Preview do PDF carregado (primeira página)
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
    
    # Inicializa containers para feedback
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    try:
        # ==============================================================================
        # ETAPA 1: Extração de texto e preparação de preview
        # ==============================================================================
        status_text.text("📄 Etapa 1/3: Extraindo texto do PDF...")
        progress_bar.progress(20)
        
        pdf_bytes = pdf_file.read()
        pages_pil = pdf_to_pil_images(pdf_bytes, dpi=300)
        texto_notas = extract_text_from_pdf(pdf_bytes, page_index=0)
        
        # ==============================================================================
        # ETAPA 2: Classificação + Extração de Normas (LLM única)
        # ==============================================================================
        status_text.text("🔍 Etapa 2/3: Classificando peça e extraindo normas...")
        progress_bar.progress(50)
        
        # Substitui placeholder no prompt unificado
        prompt_completo = classificacao_e_normas_prompt.replace("{{texto_extraido}}", texto_notas)
        
        classificacao_normas_result, classificacao_normas_metadata = classify_and_extract_norms(
            texto_notas=texto_notas,
            system_prompt=prompt_completo,
        )
        cost_logger.log_analysis(classificacao_normas_metadata, page_number=1)
        
        # ==============================================================================
        # ETAPA 3: Inferência de Normas Faltantes (LLM 2)
        # ==============================================================================
        status_text.text("🎯 Etapa 3/3: Inferindo normas faltantes...")
        progress_bar.progress(80)
        
        inferencia_prompt = f"""
Você é um especialista em normas técnicas de engenharia.

Baseado na seguinte informação:
- Tipo de peça: {classificacao_normas_result.classificacao}
- Normas encontradas: {', '.join(classificacao_normas_result.lista_normas) if classificacao_normas_result.lista_normas else 'Nenhuma'}

Identifique quais outras normas técnicas (ISO, ABNT, DIN, etc) deveriam estar aplicadas a este tipo de peça e por quê.
Considere normas de:
- Material e tratamento
- Dimensionamento e tolerâncias
- Acabamento e qualidade
- Segurança e conformidade

Retorne apenas as normas que são RECOMENDADAS e não foram encontradas no desenho.
"""
        
        inferencia_result, inferencia_metadata = infer_missing_norms(
            classificacao=classificacao_normas_result.classificacao,
            lista_normas_atuais=classificacao_normas_result.lista_normas,
            system_prompt=inferencia_prompt,
        )
        cost_logger.log_analysis(inferencia_metadata, page_number=1)
        
        progress_bar.progress(100)
        status_text.text("✅ Análise concluída com sucesso!")
        
        total_time = time.time() - start_time
        
        # Salva no session_state
        st.session_state["classification_results"] = {
            "classificacao_normas": classificacao_normas_result,
            "inferencia": inferencia_result,
            "metadata": {
                "classificacao_normas": classificacao_normas_metadata,
                "inferencia": inferencia_metadata,
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
    
    st.divider()
    st.write("## 📊 Resultados da Análise")
    
    # Imagem da peça analisada
    col_img, col_info = st.columns([1, 1])
    
    with col_img:
        st.write("#### Imagem Analisada")
        image_zoom(results["image"])
    
    with col_info:
        st.write("#### Informações da Análise")
        
        total_input_tokens = (
            results["metadata"]["classificacao_normas"].prompt_tokens +
            results["metadata"]["inferencia"].prompt_tokens
        )
        total_output_tokens = (
            results["metadata"]["classificacao_normas"].completion_tokens +
            results["metadata"]["inferencia"].completion_tokens
        )
        total_tokens = total_input_tokens + total_output_tokens
        
        col_tokens1, col_tokens2, col_tokens3 = st.columns(3)
        with col_tokens1:
            st.metric("Input Tokens", total_input_tokens)
        with col_tokens2:
            st.metric("Output Tokens", total_output_tokens)
        with col_tokens3:
            st.metric("Total Tokens", total_tokens)
        
        st.metric("Tempo Total de Processamento", f"{results['total_time']:.2f}s")
    
    st.divider()
    
    # ==============================================================================
    # Exibição simples do JSON em visual
    # ==============================================================================
    st.write("## 📋 Resultado em JSON")
    
    # JSON completo
    resultado_completo = {
        "classificacao_e_normas": results["classificacao_normas"].model_dump(),
        "normas_faltantes": results["inferencia"].model_dump(),
        "metadados": {
            "tempo_total_segundos": results["total_time"],
            "tokens": {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_tokens,
            }
        }
    }
    
    # Exibe em expandable JSON
    st.json(resultado_completo)

elif not pdf_file:
    st.info("👆 Faça o upload de um arquivo PDF para iniciar a análise")
