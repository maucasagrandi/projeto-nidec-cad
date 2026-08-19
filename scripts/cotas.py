import fitz  # PyMuPDF
import re
import statistics

def calcular_limites(marcadores_dict):
    """
    Recebe um dicionário de marcadores { 'F': 100, 'E': 250 }
    Retorna os limites numéricos que separam cada célula da grade.
    """
    if not marcadores_dict:
        return []

    # Ordena os marcadores pela coordenada (X ou Y)
    marcadores_ordenados = sorted(marcadores_dict.items(), key=lambda item: item[1])
    limites = []

    # Calcula o ponto médio entre cada marcador para criar a fronteira
    for i in range(len(marcadores_ordenados) - 1):
        id_atual, coord_atual = marcadores_ordenados[i]
        id_prox, coord_prox = marcadores_ordenados[i+1]

        ponto_medio = (coord_atual + coord_prox) / 2.0
        limites.append({"id": id_atual, "fim": ponto_medio})

    # O último marcador se estende até o limite final da página
    limites.append({"id": marcadores_ordenados[-1][0], "fim": float('inf')})

    return limites

def localizar_quadrante(x, y, limites_x, limites_y):
    """Cruza as coordenadas X e Y da cota com as malhas calculadas."""
    linha = "?"
    coluna = "?"

    # Encontra a Coluna (Eixo X)
    for limite in limites_x:
        if x <= limite["fim"]:
            coluna = limite["id"]
            break

    # Encontra a Linha (Eixo Y)
    for limite in limites_y:
        if y <= limite["fim"]:
            linha = limite["id"]
            break

    return f"{linha}{coluna}"

def processar_desenho_cad(pdf_path):
    print(f"--- Iniciando análise do arquivo: {pdf_path} ---")
    doc = fitz.open(pdf_path)
    page = doc[0] # Pega a primeira página

    # Dimensões dinâmicas da página
    largura_pagina = page.rect.width
    altura_pagina = page.rect.height

    # Zonas de Margem (Define os 8% extremos de cada borda para buscar a grade)
    ZONA_TOPO = altura_pagina * 0.08
    ZONA_BASE = altura_pagina * 0.92
    ZONA_ESQUERDA = largura_pagina * 0.08
    ZONA_DIREITA = largura_pagina * 0.92

    # Extrai todos os blocos de texto
    text_data = page.get_text("dict")

    letras_y = {}
    numeros_x = {}
    cotas_brutas = []

    # Regex para Letras (A-H), Números (1-15) e Cotas (ex: 103,9 ou 3.5±0.1)
    regex_letra = re.compile(r'^[A-H]$', re.IGNORECASE)
    regex_numero = re.compile(r'^(?:[1-9]|1[0-5])$')
    regex_cota = re.compile(r'^\(?\d+[.,]\d*(?:[±+-]\d+[.,]\d+)?\)?$')

    print("Mapeando elementos com Zonas de Exclusão (Margens)...")

    for block in text_data.get("blocks", []):
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                texto = span["text"].strip().upper() # Padroniza para maiúsculo
                if not texto: continue

                bbox = span["bbox"] # (x0, y0, x1, y1)
                centro_x = (bbox[0] + bbox[2]) / 2.0
                centro_y = (bbox[1] + bbox[3]) / 2.0

                # 1. Filtro da Grade Alfabética (Letras): Apenas laterais direita/esquerda
                if regex_letra.fullmatch(texto) and (centro_x < ZONA_ESQUERDA or centro_x > ZONA_DIREITA):
                    letras_y.setdefault(texto, []).append(centro_y)

                # 2. Filtro da Grade Numérica (Números): Apenas topo/base
                elif regex_numero.fullmatch(texto) and (centro_y < ZONA_TOPO or centro_y > ZONA_BASE):
                    numeros_x.setdefault(texto, []).append(centro_x)

                # 3. Filtro de Cotas Dimensionais: Qualquer lugar do PDF
                elif regex_cota.fullmatch(texto):
                    cotas_brutas.append({
                        "valor": texto,
                        "x": centro_x,
                        "y": centro_y,
                        "bbox": bbox
                    })

    # Calcula a posição da grade pela mediana (resolve duplicidades se houver margem em cima e embaixo)
    try:
        marcadores_y_consolidados = {letra: statistics.median(coords) for letra, coords in letras_y.items()}
        marcadores_x_consolidados = {num: statistics.median(coords) for num, coords in numeros_x.items()}
    except statistics.StatisticsError:
        print("Erro: Não foram encontrados marcadores suficientes nas margens do PDF.")
        return []

    # Cria a malha matemática
    limites_y = calcular_limites(marcadores_y_consolidados)
    limites_x = calcular_limites(marcadores_x_consolidados)

    # Mapeamento final
    resultados_finais = []
    for cota in cotas_brutas:
        quadrante = localizar_quadrante(cota["x"], cota["y"], limites_x, limites_y)
        resultados_finais.append({
            "cota": cota["valor"],
            "quadrante": quadrante,
            "bbox": cota["bbox"]
        })

    print(f"Sucesso! {len(resultados_finais)} cotas mapeadas espacialmente.\n")
    return resultados_finais

# ==========================================
# EXECUÇÃO DO SCRIPT
# ==========================================
if __name__ == "__main__":
    ARQUIVO_PDF = "19308765_rev00.pdf"

    try:
        dados_extraidos = processar_desenho_cad(ARQUIVO_PDF)

        # Exibe os resultados formatados
        if dados_extraidos:
            print("RESULTADOS:")
            print("-" * 50)
            print(f"{'QUADRANTE':<12} | {'COTA EXTRAÍDA':<20} | {'COORDENADAS (X, Y)':<20}")
            print("-" * 50)

            for item in dados_extraidos:
                x_arredondado = round(item['bbox'][0], 1)
                y_arredondado = round(item['bbox'][1], 1)
                coords = f"({x_arredondado}, {y_arredondado})"
                print(f"{item['quadrante']:<12} | {item['cota']:<20} | {coords:<20}")

    except Exception as e:
        print(f"Erro ao processar o PDF: {e}")