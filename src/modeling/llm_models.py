import base64
import os
import logging
import time
import json
from dataclasses import dataclass
from typing import Tuple, List
from datetime import datetime
from pydantic import BaseModel, Field
import google.genai as genai
from google.genai import types

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
    lista_normas: List[str] = Field(description="Lista de normas encontradas")
    justificativas: List[str] = Field(description="Evidência textual de cada norma")


class ClassificacaoENormasOutput(BaseModel):
    """Estrutura de saída unificada: classificação da peça + extração de normas"""
    classificacao: str = Field(description="Tipo da peça identificado")
    justificativa_classificacao: str = Field(description="Trecho ou evidência textual que identifica o tipo da peça")
    lista_normas: List[str] = Field(description="Lista de normas encontradas")
    justificativas_normas: List[str] = Field(description="Evidência textual de cada norma")


class NormasFaltantesOutput(BaseModel):
    """Estrutura de saída para inferência de normas faltantes"""
    normas_sugeridas: List[str] = Field(description="Lista de normas recomendadas")
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
logger.info(f"Inicializando cliente Gemini...")
logger.info(f"Projeto: {GCP_PROJECT}, Região: {GCP_LOCATION}, Modelo: {MODEL_ID}")

client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)
logger.info("✅ Cliente Gemini inicializado")


# ==============================================================================
# Funções LLM com Structured Output
# ==============================================================================

def classify_part_from_image(
    image_base64: str,
    system_prompt: str,
    model: str = MODEL_ID,
) -> Tuple[ClassificacaoOutput, AnalysisMetadata]:
    """Classifica peça com structured output garantido."""
    start_time = time.time()
    
    logger.info(f"Enviando classificação para {model}...")
    
    image_data = base64.b64decode(image_base64)
    
    response = client.models.generate_content(
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
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"✅ Classificação: {parsed.classificacao}")
    return parsed, metadata


def extract_norms_from_text(
    texto_notas: str,
    system_prompt: str,
    model: str = MODEL_ID,
) -> Tuple[NormasOutput, AnalysisMetadata]:
    """Extrai normas com structured output garantido."""
    start_time = time.time()
    
    logger.info(f"Enviando extração de normas para {model}...")
    
    response = client.models.generate_content(
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
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"✅ Normas: {len(parsed.lista_normas)} encontradas")
    return parsed, metadata


def classify_and_extract_norms(
    texto_notas: str,
    system_prompt: str,
    model: str = MODEL_ID,
) -> Tuple[ClassificacaoENormasOutput, AnalysisMetadata]:
    """Classifica peça e extrai normas em uma única chamada LLM (texto)."""
    start_time = time.time()
    
    logger.info(f"Enviando classificação + normas (unificado) para {model}...")
    
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=system_prompt),
        ],
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
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"✅ Classificação: {parsed.classificacao} | Normas: {len(parsed.lista_normas)} encontradas")
    return parsed, metadata


def infer_missing_norms(
    classificacao: str,
    lista_normas_atuais: List[str],
    system_prompt: str,
    model: str = MODEL_ID,
) -> Tuple[NormasFaltantesOutput, AnalysisMetadata]:
    """Infere normas faltantes com structured output garantido."""
    start_time = time.time()
    
    logger.info(f"Enviando inferência para {model}...")
    
    normas_str = ", ".join(lista_normas_atuais) if lista_normas_atuais else "Nenhuma"
    prompt = f"{system_prompt}\n\nPeça: {classificacao}\nNormas atuais: {normas_str}"
    
    response = client.models.generate_content(
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
        timestamp=datetime.now().isoformat()
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
) -> Tuple[str, AnalysisMetadata]:
    """Compara duas páginas CAD (modo texto)."""
    start_time = time.time()
    
    logger.info(f"Enviando comparação para {model}...")
    
    image1_data = base64.b64decode(image1_base64)
    image2_data = base64.b64decode(image2_base64)
    
    response = client.models.generate_content(
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
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"✅ Comparação recebida")
    return response.text, metadata
