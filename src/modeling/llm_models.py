import google.genai as genai
import base64
import os
import logging
import time
from dataclasses import dataclass
from typing import Tuple
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Metadados da análise de CAD"""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    timestamp: str

# ==============================================================================
# Cliente Gemini via Vertex AI (Simples)
# ==============================================================================

logger.info(f"Inicializando cliente Gemini...")
logger.info(f"Projeto: {GCP_PROJECT}, Região: {GCP_LOCATION}, Modelo: {MODEL_ID}")

try:
    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )
    logger.info("✅ Cliente Gemini inicializado com sucesso")
except Exception as e:
    logger.error(f"❌ Erro ao inicializar cliente: {e}")
    raise


def compare_cad_pages(
    image1_base64: str,
    image2_base64: str,
    system_prompt: str,
    model: str = MODEL_ID,
    max_tokens: int = 32768,
) -> Tuple[str, AnalysisMetadata]:
    """
    Envia duas imagens de páginas CAD para o Gemini via Vertex AI e retorna
    a análise comparativa com metadados de uso.

    Args:
        image1_base64: Imagem da página do PDF original, codificada em base64.
        image2_base64: Imagem da página do PDF revisado, codificada em base64.
        system_prompt: Prompt de sistema com as instruções de análise.
        model: Modelo Gemini (padrão: gemini-2.5-flash-image).
        max_tokens: Número máximo de tokens na resposta.

    Returns:
        Tuple com (texto_análise, AnalysisMetadata com tokens e latência)
    
    Raises:
        Exception: Se houver erro na chamada à API.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Enviando análise para {model}...")
        logger.info(f"Imagem 1: {len(image1_base64)} chars | Imagem 2: {len(image2_base64)} chars")
        
        # Decodifica as imagens base64
        image1_data = base64.b64decode(image1_base64)
        image2_data = base64.b64decode(image2_base64)
        
        # Monta o conteúdo com as duas imagens
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image1_data).decode('utf-8'),
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image2_data).decode('utf-8'),
                        }
                    },
                    {
                        "text": (
                            f"{system_prompt}\n\n"
                            "A primeira imagem é o desenho CAD ORIGINAL. "
                            "A segunda imagem é o desenho CAD REVISADO. "
                            "Identifique e liste todas as divergências entre eles."
                        )
                    },
                ],
            }
        ]
        
        # Chama o modelo
        response = client.models.generate_content(
            model=f"projects/{GCP_PROJECT}/locations/{GCP_LOCATION}/publishers/google/models/{model}",
            contents=contents,
        )
        
        logger.info("✅ Resposta recebida")
        
        result_text = response.text
        
        # Calcula latência
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extrai tokens da resposta
        usage_metadata = response.usage_metadata
        total_tokens = usage_metadata.total_token_count
        prompt_tokens = usage_metadata.prompt_token_count
        completion_tokens = usage_metadata.candidates_token_count
        
        logger.info(f"Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}")
        logger.info(f"Latência: {latency_ms:.2f}ms")
        
        # Cria metadados
        metadata = AnalysisMetadata(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model=model,
            timestamp=datetime.now().isoformat()
        )
        
        return result_text, metadata

    except Exception as e:
        logger.error(f"❌ Erro ao analisar CAD: {type(e).__name__}: {str(e)}")
        raise
