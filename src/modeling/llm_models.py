import base64
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Pydantic Models para Structured Output
# ==============================================================================

class ClassificacaoOutput(BaseModel):
    """Estrutura de saída para classificação de peça"""
    classificacao: str = Field(description="Tipo da peça identificado")
    justificativa: str = Field(description="Justificativa baseada em características visuais")


class NormasOutput(BaseModel):
    """Estrutura de saída para extração de normas"""
    lista_normas: list[str] = Field(description="Lista de normas encontradas")
    justificativas: list[str] = Field(description="Evidência textual de cada norma")


class HeaderOutput(BaseModel):
    """Customer-facing drawing header extracted from the revised PDF."""

    drawing_number: str | None = Field(default=None, description="No. at drawing block")
    title: str | None = Field(default=None, description="TITLE, DOCUMENT TYPE at drawing block")
    compressor_series_code: str | None = Field(
        default=None,
        description="Compressor series explicitly present in the drawing; null when external lookup is required",
    )
    cr: str | None = Field(default=None, description="ECM or ECAM value at drawing block")
    classification: str | None = Field(default=None, description="Semantic part classification")
    last_revision_date: str | None = Field(
        default=None,
        description="DATE on the latest row or column of the revision table",
    )


class DrawingBlockOutput(BaseModel):
    """Literal transcription of the revised drawing title block."""

    materials: list[str] = Field(default_factory=list, description="All MATERIAL values")
    material_code: str | None = Field(default=None, description="MATERIAL CODE or CODE value")
    drawn_by: str | None = Field(default=None, description="DRAWN value")
    approved_by: str | None = Field(default=None, description="APP. or APPROVED value")
    drawing_code_ecm: str | None = Field(default=None, description="ECM value")
    date: str | None = Field(default=None, description="DATE at drawing block")
    name_and_document_type: str | None = Field(default=None, description="TITLE, DOCUMENT TYPE value")
    general_tolerance: str | None = Field(default=None, description="GEN. TOL. value")
    angular_tolerance: str | None = Field(default=None, description="ANG. TOL. value")
    scale: str | None = Field(default=None, description="SCALE value")
    unit: str | None = Field(default=None, description="UNIT value")
    replace: str | None = Field(default=None, description="REPLACE value")
    number: str | None = Field(default=None, description="No. value")


class ClassificacaoENormasOutput(BaseModel):
    """Unified multimodal drawing header, classification and standards output."""

    header: HeaderOutput
    drawing_block: DrawingBlockOutput
    classificacao: str = Field(description="Tipo da peça identificado")
    justificativa_classificacao: str = Field(description="Trecho ou evidência textual que identifica o tipo da peça")
    lista_normas: list[str] = Field(description="Lista de normas encontradas")
    justificativas_normas: list[str] = Field(description="Evidência textual de cada norma")


class NormasFaltantesOutput(BaseModel):
    """Estrutura de saída para inferência de normas faltantes"""
    normas_sugeridas: list[str] = Field(description="Lista de normas recomendadas")
    reasoning: str = Field(description="Explicação técnica")
    confianca: float = Field(description="Nível de confiança (0.0 a 1.0)")


# ==============================================================================
# Configuração
# ==============================================================================
GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "acim-global-data-lake-sandbox")
GCP_LOCATION = os.getenv("GCP_REGION", "global")
MODEL_ID = "gemini-3.5-flash"

# ==============================================================================
# Dataclass para armazenar metadados de uso
# ==============================================================================
@dataclass
class AnalysisMetadata:
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    timestamp: str

# ==============================================================================
# Cliente Gemini
# ==============================================================================
client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Initialize Gemini only when an analysis call is actually made."""

    global client
    if client is None:
        logger.info("Inicializando cliente Gemini...")
        logger.info(f"Projeto: {GCP_PROJECT}, Região: {GCP_LOCATION}, Modelo: {MODEL_ID}")
        client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
        logger.info("✅ Cliente Gemini inicializado")
    return client


# ==============================================================================
# Funções LLM com Structured Output
# ==============================================================================

def classify_part_from_image(
    image_base64: str,
    system_prompt: str,
    model: str = MODEL_ID,
) -> tuple[ClassificacaoOutput, AnalysisMetadata]:
    """Classifica peça com structured output garantido."""
    start_time = time.time()
    
    logger.info(f"Enviando classificação para {model}...")
    
    image_data = base64.b64decode(image_base64)
    
    response = _get_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_data, mime_type="image/png"),
            types.Part.from_text(text=system_prompt),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificacaoOutput,
        ),
    )
    
    parsed = ClassificacaoOutput.model_validate_json(response.text)
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    usage = response.usage_metadata
    metadata = AnalysisMetadata(
        total_tokens=usage.total_token_count,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        latency_ms=latency_ms,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    logger.info(f"✅ Classificação: {parsed.classificacao}")
    return parsed, metadata


def extract_norms_from_text(
    texto_notas: str,
    system_prompt: str,
    model: str = MODEL_ID,
) -> tuple[NormasOutput, AnalysisMetadata]:
    """Extrai normas com structured output garantido."""
    start_time = time.time()
    
    logger.info(f"Enviando extração de normas para {model}...")
    
    response = _get_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=f"{system_prompt}\n\nTexto:\n{texto_notas}"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=NormasOutput,
        ),
    )
    
    parsed = NormasOutput.model_validate_json(response.text)
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    usage = response.usage_metadata
    metadata = AnalysisMetadata(
        total_tokens=usage.total_token_count,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        latency_ms=latency_ms,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    logger.info(f"✅ Normas: {len(parsed.lista_normas)} encontradas")
    return parsed, metadata


def classify_and_extract_norms(
    texto_notas: str,
    system_prompt: str,
    model: str = MODEL_ID,
    pdf_bytes: bytes | None = None,
) -> tuple[ClassificacaoENormasOutput, AnalysisMetadata]:
    """Extract drawing metadata, classify the part and find standards in one call."""
    start_time = time.time()
    
    logger.info(f"Enviando carimbo + classificação + normas (multimodal) para {model}...")

    contents = []
    if pdf_bytes:
        contents.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
    contents.append(types.Part.from_text(text=system_prompt))
    
    response = _get_client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificacaoENormasOutput,
        ),
    )
    
    parsed = ClassificacaoENormasOutput.model_validate_json(response.text)
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    usage = response.usage_metadata
    metadata = AnalysisMetadata(
        total_tokens=usage.total_token_count,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        latency_ms=latency_ms,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    logger.info(f"✅ Classificação: {parsed.classificacao} | Normas: {len(parsed.lista_normas)} encontradas")
    return parsed, metadata


def infer_missing_norms(
    classificacao: str,
    lista_normas_atuais: list[str],
    system_prompt: str,
    model: str = MODEL_ID,
) -> tuple[NormasFaltantesOutput, AnalysisMetadata]:
    """Infere normas faltantes com structured output garantido."""
    start_time = time.time()
    
    logger.info(f"Enviando inferência para {model}...")
    
    normas_str = ", ".join(lista_normas_atuais) if lista_normas_atuais else "Nenhuma"
    prompt = f"{system_prompt}\n\nPeça: {classificacao}\nNormas atuais: {normas_str}"
    
    response = _get_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=NormasFaltantesOutput,
        ),
    )
    
    parsed = NormasFaltantesOutput.model_validate_json(response.text)
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    usage = response.usage_metadata
    metadata = AnalysisMetadata(
        total_tokens=usage.total_token_count,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        latency_ms=latency_ms,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    logger.info(f"✅ Sugestões: {len(parsed.normas_sugeridas)} normas")
    return parsed, metadata


# ==============================================================================
# Funções Utilitárias
# ==============================================================================

def extract_text_from_pdf(pdf_bytes: bytes, page_index: int = 0) -> str:
    """Extrai texto do PDF usando PyMuPDF."""
    import fitz
    
    logger.info(f"Extraindo texto da página {page_index + 1}...")
    
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_index >= len(pdf_doc):
        page_index = 0
    
    texto = pdf_doc[page_index].get_text()
    pdf_doc.close()
    
    logger.info(f"✅ Texto extraído: {len(texto)} caracteres")
    return texto


def compare_cad_pages(
    image1_base64: str,
    image2_base64: str,
    system_prompt: str,
    model: str = "gemini-3.5-flash",
    max_tokens: int = 32768,
) -> tuple[str, AnalysisMetadata]:
    """Compara duas páginas CAD (modo texto)."""
    start_time = time.time()
    
    logger.info(f"Enviando comparação para {model}...")
    
    image1_data = base64.b64decode(image1_base64)
    image2_data = base64.b64decode(image2_base64)
    
    response = _get_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image1_data, mime_type="image/png"),
            types.Part.from_bytes(data=image2_data, mime_type="image/png"),
            types.Part.from_text(text=f"{system_prompt}\n\nPrimeira: ORIGINAL. Segunda: REVISADA."),
        ],
    )
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    usage = response.usage_metadata
    metadata = AnalysisMetadata(
        total_tokens=usage.total_token_count,
        prompt_tokens=usage.prompt_token_count,
        completion_tokens=usage.candidates_token_count,
        latency_ms=latency_ms,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    logger.info("✅ Comparação recebida")
    return response.text, metadata
