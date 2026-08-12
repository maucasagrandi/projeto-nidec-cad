import fitz  # PyMuPDF
import re
import statistics
import os

from PIL import Image, ImageDraw, ImageFont


def resolver_caminho_pdf(caminho_relativo):
    """
    Localiza o PDF independentemente de onde o script é executado.
    Remove qualquer prefixo "../" do caminho informado e procura o
    restante do caminho a partir do diretório deste script e de cada
    uma das pastas pai (subindo até 6 níveis), até encontrar o arquivo.
    """
    partes = caminho_relativo.replace("\\", "/").split("/")
    partes_limpas = [p for p in partes if p not in ("", "..", ".")]
    sufixo = os.path.join(*partes_limpas)

    base = os.path.dirname(os.path.abspath(__file__))
    candidatos = [caminho_relativo]
    for _ in range(6):
        candidatos.append(os.path.join(base, sufixo))
        candidatos.append(os.path.join(base, os.path.basename(sufixo)))
        novo_base = os.path.dirname(base)
        if novo_base == base:
            break
        base = novo_base

    for candidato in candidatos:
        if os.path.isfile(candidato):
            print(f"Arquivo encontrado em: {os.path.abspath(candidato)}")
            return candidato

    print(f"Diretório deste script: {os.path.dirname(os.path.abspath(__file__))}")
    print("Nenhum dos caminhos abaixo foi encontrado:")
    for candidato in candidatos:
        print(f"  - {os.path.abspath(candidato)}")
    raise FileNotFoundError(
        "PDF não encontrado. Ajuste ARQUIVO_PDF ou verifique o caminho acima."
    )


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
        id_prox, coord_prox = marcadores_ordenados[i + 1]

        ponto_medio = (coord_atual + coord_prox) / 2.0
        limites.append({"id": id_atual, "fim": ponto_medio})

    # O último marcador se estende até o limite final da página
    limites.append({"id": marcadores_ordenados[-1][0], "fim": float("inf")})

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
    pdf_path = resolver_caminho_pdf(pdf_path)
    doc = fitz.open(pdf_path)
    page = doc[0]  # Pega a primeira página

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
    regex_letra = re.compile(r"^[A-H]$", re.IGNORECASE)
    regex_numero = re.compile(r"^(?:[1-9]|1[0-5])$")
    regex_cota = re.compile(r"^\(?\d+[.,]\d*(?:[±+-]\d+[.,]\d+)?\)?$")

    print("Mapeando elementos com Zonas de Exclusão (Margens)...")

    for block in text_data.get("blocks", []):
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                texto = span["text"].strip().upper()  # Padroniza para maiúsculo
                if not texto:
                    continue

                bbox = span["bbox"]  # (x0, y0, x1, y1)
                centro_x = (bbox[0] + bbox[2]) / 2.0
                centro_y = (bbox[1] + bbox[3]) / 2.0

                # 1. Filtro da Grade Alfabética (Letras): Apenas laterais direita/esquerda
                if regex_letra.fullmatch(texto) and (
                    centro_x < ZONA_ESQUERDA or centro_x > ZONA_DIREITA
                ):
                    letras_y.setdefault(texto, []).append(centro_y)

                # 2. Filtro da Grade Numérica (Números): Apenas topo/base
                elif regex_numero.fullmatch(texto) and (
                    centro_y < ZONA_TOPO or centro_y > ZONA_BASE
                ):
                    numeros_x.setdefault(texto, []).append(centro_x)

                # 3. Filtro de Cotas Dimensionais: Qualquer lugar do PDF
                elif regex_cota.fullmatch(texto):
                    cotas_brutas.append(
                        {
                            "valor": texto,
                            "x": centro_x,
                            "y": centro_y,
                            "bbox": bbox,
                        }
                    )

    # Calcula a posição da grade pela mediana (resolve duplicidades se houver margem em cima e embaixo)
    try:
        marcadores_y_consolidados = {
            letra: statistics.median(coords) for letra, coords in letras_y.items()
        }
        marcadores_x_consolidados = {
            num: statistics.median(coords) for num, coords in numeros_x.items()
        }
    except statistics.StatisticsError:
        print("Erro: Não foram encontrados marcadores suficientes nas margens do PDF.")
        return [], pdf_path

    # Cria a malha matemática
    limites_y = calcular_limites(marcadores_y_consolidados)
    limites_x = calcular_limites(marcadores_x_consolidados)

    # Mapeamento final
    resultados_finais = []
    for cota in cotas_brutas:
        quadrante = localizar_quadrante(cota["x"], cota["y"], limites_x, limites_y)
        resultados_finais.append(
            {
                "cota": cota["valor"],
                "quadrante": quadrante,
                "bbox": cota["bbox"],
            }
        )

    print(f"Sucesso! {len(resultados_finais)} cotas mapeadas espacialmente.\n")
    return resultados_finais, pdf_path


def _carregar_fonte(tamanho, negrito=False):
    candidatos = (
        "arialbd.ttf" if negrito else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if negrito else "C:/Windows/Fonts/arial.ttf",
    )
    for caminho in candidatos:
        try:
            return ImageFont.truetype(caminho, size=tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def gerar_imagem_crops_cotas(
    pdf_path,
    resultados,
    output_path="cotas_crops.png",
    dpi=200,
    padding_pt=30.0,
    colunas=6,
):
    """
    Gera uma única imagem (contact sheet) contendo o recorte de cada cota
    detectada, rotulado com o valor da cota e o quadrante correspondente.

    O padding é generoso porque a tolerância (ex: "+0,1/-0,1") costuma ser
    um span de texto separado, posicionado ao lado/acima do valor principal
    capturado pela regex — sem folga suficiente, ela fica de fora do recorte.
    """
    if not resultados:
        print("Nenhuma cota para recortar — imagem não gerada.")
        return None

    doc = fitz.open(pdf_path)
    page = doc[0]
    escala = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(escala, escala))
    pagina_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()

    largura_celula = 260
    altura_crop = 130
    altura_legenda = 34
    altura_celula = altura_crop + altura_legenda
    margem = 10

    linhas = (len(resultados) + colunas - 1) // colunas
    sheet = Image.new(
        "RGB",
        (colunas * largura_celula, linhas * altura_celula + margem),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    fonte_legenda = _carregar_fonte(16, negrito=True)

    for indice, item in enumerate(resultados):
        x0, y0, x1, y1 = item["bbox"]
        pad = padding_pt
        rect_px = (
            max(0, int(round((x0 - pad) * escala))),
            max(0, int(round((y0 - pad) * escala))),
            min(pagina_img.width, int(round((x1 + pad) * escala))),
            min(pagina_img.height, int(round((y1 + pad) * escala))),
        )
        crop = pagina_img.crop(rect_px)

        max_w, max_h = largura_celula - 10, altura_crop
        escala_fit = min(max_w / crop.width, max_h / crop.height)
        # Amplia crops pequenos (texto minúsculo no PDF original) para
        # manter a legibilidade, e reduz apenas os que excedem a célula.
        escala_fit = min(escala_fit, 2.5)
        novo_w = max(1, int(round(crop.width * escala_fit)))
        novo_h = max(1, int(round(crop.height * escala_fit)))
        crop = crop.resize((novo_w, novo_h), Image.Resampling.LANCZOS)

        coluna = indice % colunas
        linha = indice // colunas
        cell_x = coluna * largura_celula
        cell_y = linha * altura_celula + margem

        crop_x = cell_x + (largura_celula - crop.width) // 2
        crop_y = cell_y + (altura_crop - crop.height) // 2
        sheet.paste(crop, (crop_x, crop_y))

        draw.rectangle(
            (cell_x + 2, cell_y, cell_x + largura_celula - 2, cell_y + altura_crop),
            outline=(180, 180, 180),
            width=1,
        )

        legenda = f"{item['cota']}  [{item['quadrante']}]"
        legenda_bbox = draw.textbbox((0, 0), legenda, font=fonte_legenda)
        legenda_w = legenda_bbox[2] - legenda_bbox[0]
        legenda_x = cell_x + (largura_celula - legenda_w) // 2
        legenda_y = cell_y + altura_crop + 4
        draw.text((legenda_x, legenda_y), legenda, fill=(30, 30, 30), font=fonte_legenda)

    sheet.save(output_path)
    print(f"Imagem com os crops de cada cota salva em: {os.path.abspath(output_path)}")
    return output_path


def gerar_imagem_pagina_marcada(
    pdf_path,
    resultados,
    output_path="cotas_marcadas.png",
    dpi=200,
    padding_pt=30.0,
):
    """
    Gera a imagem da página inteira com um retângulo desenhado sobre a
    região exata de cada crop (mesma região usada em gerar_imagem_crops_cotas),
    rotulado com o valor da cota e o quadrante correspondente.
    """
    if not resultados:
        print("Nenhuma cota para marcar — imagem não gerada.")
        return None

    doc = fitz.open(pdf_path)
    page = doc[0]
    escala = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(escala, escala))
    pagina_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
    doc.close()

    draw = ImageDraw.Draw(pagina_img)
    fonte_rotulo = _carregar_fonte(max(12, int(round(dpi / 15))), negrito=True)
    cor = (220, 30, 30)

    for indice, item in enumerate(resultados):
        x0, y0, x1, y1 = item["bbox"]
        pad = padding_pt
        rect_px = (
            max(0, int(round((x0 - pad) * escala))),
            max(0, int(round((y0 - pad) * escala))),
            min(pagina_img.width, int(round((x1 + pad) * escala))),
            min(pagina_img.height, int(round((y1 + pad) * escala))),
        )
        draw.rectangle(rect_px, outline=cor, width=2)

        rotulo = f"{item['cota']} [{item['quadrante']}]"
        rot_bbox = draw.textbbox((0, 0), rotulo, font=fonte_rotulo)
        rot_w = rot_bbox[2] - rot_bbox[0]
        rot_h = rot_bbox[3] - rot_bbox[1]

        rot_x = rect_px[0]
        rot_y = rect_px[1] - rot_h - 6
        if rot_y < 0:
            rot_y = rect_px[3] + 4

        fundo = (rot_x - 2, rot_y - 2, rot_x + rot_w + 2, rot_y + rot_h + 2)
        draw.rectangle(fundo, fill=(255, 255, 255), outline=cor, width=1)
        draw.text((rot_x, rot_y), rotulo, fill=cor, font=fonte_rotulo)

    pagina_img.save(output_path)
    print(f"Imagem da página com os crops marcados salva em: {os.path.abspath(output_path)}")
    return output_path


# ==========================================
# EXECUÇÃO DO SCRIPT
# ==========================================
if __name__ == "__main__":
    ARQUIVO_PDF = "CAD_Review_Test_Battery_V1/2. Comparison Analysis/42/13751188_REV_0_draw_1.pdf"

    try:
        dados_extraidos, pdf_path_resolvido = processar_desenho_cad(ARQUIVO_PDF)

        # Exibe os resultados formatados
        if dados_extraidos:
            print("RESULTADOS:")
            print("-" * 50)
            print(f"{'QUADRANTE':<12} | {'COTA EXTRAÍDA':<20} | {'COORDENADAS (X, Y)':<20}")
            print("-" * 50)

            for item in dados_extraidos:
                x_arredondado = round(item["bbox"][0], 1)
                y_arredondado = round(item["bbox"][1], 1)
                coords = f"({x_arredondado}, {y_arredondado})"
                print(f"{item['quadrante']:<12} | {item['cota']:<20} | {coords:<20}")

            # Gera uma imagem única com o recorte de cada cota detectada,
            # rotulada com o valor e o quadrante correspondente.
            output_path_crops = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cotas_crops.png"
            )
            gerar_imagem_crops_cotas(pdf_path_resolvido, dados_extraidos, output_path=output_path_crops)

            # Gera a página inteira com cada crop marcado (retângulo + rótulo).
            output_path_marcado = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cotas_marcadas.png"
            )
            gerar_imagem_pagina_marcada(
                pdf_path_resolvido, dados_extraidos, output_path=output_path_marcado
            )

    except Exception as e:
        print(f"Erro ao processar o PDF: {e}")
