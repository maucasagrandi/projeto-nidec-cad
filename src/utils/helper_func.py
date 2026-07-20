import fitz  # PyMuPDF
import base64
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# ==============================================================================
# PDF → lista de imagens PNG em memória (base64)
# ==============================================================================

def pdf_to_images_base64(pdf_bytes: bytes, dpi: int = 200) -> list[str]:
    """
    Converte todas as páginas de um PDF em imagens PNG e retorna uma lista
    de strings base64, uma por página.

    Args:
        pdf_bytes: Conteúdo binário do arquivo PDF.
        dpi: Resolução de renderização. 200 é um bom equilíbrio entre
             qualidade e tamanho para análise por LLM.

    Returns:
        Lista de strings base64, cada uma representando uma página em PNG.
    """
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_b64 = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        # matrix aumenta a resolução via fator de escala baseado no DPI
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        images_b64.append(b64)

    document.close()
    return images_b64


def pdf_to_pil_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """
    Converte todas as páginas de um PDF em objetos PIL.Image.

    Args:
        pdf_bytes: Conteúdo binário do arquivo PDF.
        dpi: Resolução de renderização.

    Returns:
        Lista de objetos PIL.Image, um por página.
    """
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        png_bytes = pix.tobytes("png")
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        images.append(img)

    document.close()
    return images


# ==============================================================================
# Diff visual entre duas páginas com OpenCV
# ==============================================================================

def compute_visual_diff(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Calcula a diferença visual entre duas imagens de páginas CAD.
    Retorna a imagem da segunda página com os contornos das diferenças
    marcados em vermelho.

    Args:
        img1: Imagem da página original (PIL.Image).
        img2: Imagem da página revisada (PIL.Image).

    Returns:
        PIL.Image com as diferenças marcadas em vermelho.
    """
    # Converte para arrays numpy BGR (OpenCV)
    arr1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    arr2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    # Redimensiona img2 para o tamanho de img1, se necessário
    if arr1.shape != arr2.shape:
        arr2 = cv2.resize(arr2, (arr1.shape[1], arr1.shape[0]))

    # Diferença absoluta
    diff = cv2.absdiff(arr1, arr2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Aplica blur para suavizar ruído de compressão/renderização
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)

    # Threshold mais sensível para captar mudanças sutis em CAD
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

    # Remove ruído morfológico e expande regiões detectadas
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=3)

    # Encontra contornos das regiões modificadas
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Espessura proporcional ao tamanho da imagem
    h, w = arr2.shape[:2]
    thickness = max(3, int(min(h, w) / 300))
    padding = max(5, int(min(h, w) / 200))

    # Desenha retângulos vermelhos ao redor de cada região diferente
    output = arr2.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30:  # limiar mais sensível
            x, y, w_r, h_r = cv2.boundingRect(cnt)
            # Adiciona padding ao redor da região
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_r = min(w - x, w_r + 2 * padding)
            h_r = min(h - y, h_r + 2 * padding)
            cv2.rectangle(output, (x, y), (x + w_r, y + h_r), (0, 0, 255), thickness)

    # Converte de volta para PIL RGB
    result = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    return result


def count_diff_regions(img1: Image.Image, img2: Image.Image) -> int:
    """
    Retorna o número de regiões com diferença visual entre duas imagens.
    Útil para pré-filtrar páginas que realmente mudaram.
    """
    arr1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    arr2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    if arr1.shape != arr2.shape:
        arr2 = cv2.resize(arr2, (arr1.shape[1], arr1.shape[0]))

    diff = cv2.absdiff(arr1, arr2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel, iterations=3)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) > 30)


# ==============================================================================
# Utilidades gerais
# ==============================================================================

def pil_to_base64(img: Image.Image) -> str:
    """Converte um PIL.Image para string base64 PNG."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ==============================================================================
# Otimização de imagens para LLM
# ==============================================================================

def compress_png_for_llm(img: Image.Image, quality: int = 85) -> str:
    """
    Comprime uma imagem PIL em PNG otimizado para reduzir tokens no Gemini.
    Remove metadados e aplica máxima compressão.
    
    Args:
        img: PIL.Image para comprimir
        quality: Nível de compressão (1-100, padrão 85 = bom balanço)
    
    Returns:
        String base64 da imagem comprimida
    """
    buffered = BytesIO()
    # Salva com otimização e máxima compressão
    img.save(
        buffered,
        format="PNG",
        optimize=True,          # Remove metadados
        compress_level=9        # Máxima compressão (0-9)
    )
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
