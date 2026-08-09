"""
Artefato de validacao — pasta 41 (13358002_REV_8_draw_2.pdf).

1. Detecta todos os CANDIDATOS a quadro GD&T (reconstrucao vetorial,
   Topico 7 do prompt_classification.md) com o endpoint_tolerance
   corrigido (4.5 — ver src/utils/gdt_detector.py).
2. Para cada candidato, recorta a primeira celula (onde fica o simbolo)
   e classifica contra os 3 templates reais do cliente na raiz do repo,
   comparando SEMPRE contra os 3 ao mesmo tempo (matchTemplate). A classe
   reportada e a que tiver o maior score entre os 3, desde que passe
   min_score e min_margin:
       cota1.png -> rotulo "generic_circle" (circulo simples, controle negativo)
       cota2.png -> rotulo "profile"        (arco)
       cota3.png -> rotulo "position"       (circulo com cruz / mira)
3. Gera uma unica imagem anotada (pasta41_candidatos_vs_validados.png):
       azul   = candidato a quadro (estrutura geometrica encontrada)
       verde  = candidato cuja 1a celula foi VALIDADA contra um dos
                3 templates (score >= min_score e margem >= min_margin)

Reexecutar: python test_outputs/pasta41_candidates_vs_cotas/pasta41_candidates_vs_validated.py
(funciona a partir de qualquer cwd, os caminhos sao resolvidos via PROJECT_ROOT)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.gdt_cell_classifier import GDT_CELL_INSET_FRACTION, GDT_TARGET_SIZE,TEST_NUMBER

# Raiz do projeto = 2 niveis acima deste arquivo (test_outputs/pasta41_candidates_vs_cotas/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import ImageDraw, ImageFont

from src.utils.gdt_detector import GdtFrameDetector
from src.utils.gdt_template_matcher import TemplateSpec, render_page
from src.utils.gdt_cell_classifier import (
    CandidateCell,
    prepare_templates_canonical,
    crop_cell_interior,
    build_canonical_forms,
    classify_cell,
)

STR_TEST_NUMBER = str(TEST_NUMBER)
STR_GDT_TARGET_SIZE = str(GDT_TARGET_SIZE)
STR_GDT_CELL_INSET_FRACTION = str(GDT_CELL_INSET_FRACTION)
 
PDF_PATH = PROJECT_ROOT / "CAD Comparison Analysis V1.0" / "Sample" / "41" / "13358002_REV_8_draw_2.pdf"
PAGE_INDEX = 0
OUT_PATH = str(Path(__file__).parent / f"teste {STR_TEST_NUMBER}- Size {STR_GDT_TARGET_SIZE} - Fraction {STR_GDT_CELL_INSET_FRACTION}.png")


MIN_SCORE = 0.55
MIN_MARGIN = 0.12

# ------------------------------------------------------------------------
# 1. Detecta candidatos a quadro GD&T
# ------------------------------------------------------------------------
pdf_bytes = PDF_PATH.read_bytes()

detector = GdtFrameDetector()  # endpoint_tolerance=4.5 (default corrigido)
candidates = detector.detect_frames(pdf_bytes, page_index=PAGE_INDEX)
print(f"Candidatos a quadro GD&T detectados: {len(candidates)}")

# ------------------------------------------------------------------------
# 2. Prepara os 3 templates reais do cliente (cota1/cota2/cota3.png)
# ------------------------------------------------------------------------
template_specs = [
    TemplateSpec(name="generic_circle", path=str(PROJECT_ROOT / "cota5.png"))
]
templates_forms = prepare_templates_canonical(template_specs)
print(f"Templates preparados: {list(templates_forms.keys())}")

rendered = render_page(pdf_bytes, PAGE_INDEX, dpi=detector.crop_dpi)

# ------------------------------------------------------------------------
# 3. Classifica a 1a celula (simbolo) de cada candidato contra os templates
# ------------------------------------------------------------------------
results = []
for cand in candidates:
    symbol_cell = CandidateCell(
        cell_id=cand.candidate_id,
        bbox=cand.symbol_bbox,
        frame_bbox=cand.frame_bbox,
        cell_index=0,
        num_cells_in_frame=len(cand.cells),
        is_first_cell=True,
    )

    interior = crop_cell_interior(rendered.gray, cand.symbol_bbox, rendered.zoom)
    if interior is None:
        results.append((cand, None))
        continue

    cell_forms = build_canonical_forms(interior)
    classification = classify_cell(
        symbol_cell, cell_forms, templates_forms,
        min_score=MIN_SCORE, min_margin=MIN_MARGIN,
    )
    results.append((cand, classification))

validated = [(c, cl) for c, cl in results if cl is not None and cl.accepted]
print(f"Candidatos validados contra cota1/cota2/cota3: {len(validated)} de {len(candidates)}")
for cand, cl in validated:
    print(f"  {cand.candidate_id}: classe={cl.best.template_name} "
          f"score={cl.best.score:.3f} margin={cl.margin:.3f} "
          f"bbox={cand.frame_bbox.to_list()}")

print("\nNao validados (score/margem insuficiente ou sem crop):")
for cand, cl in results:
    if cl is None:
        print(f"  {cand.candidate_id}: sem crop valido")
    elif not cl.accepted:
        best = cl.best
        if best is not None:
            print(f"  {cand.candidate_id}: melhor={best.template_name} "
                  f"score={best.score:.3f} margin={cl.margin:.3f}")
        else:
            print(f"  {cand.candidate_id}: sem score")

# ------------------------------------------------------------------------
# 4. Gera imagem anotada: azul = candidato, verde = validado
# ------------------------------------------------------------------------
img = detector._last_page_image.copy()
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 16)
except (OSError, IOError):
    font = ImageFont.load_default()


def draw_label(xy, text, bg):
    x, y = xy
    try:
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([x, y - th - 2, x + tw + 4, y], fill=bg)
    draw.text((x + 2, y - th - 1), text, fill="white", font=font)


validated_ids = {cand.candidate_id for cand, cl in validated}

for cand, cl in results:
    px = detector._bbox_to_pixels(cand.frame_bbox)
    is_validated = cand.candidate_id in validated_ids

    color = (0, 170, 0) if is_validated else (30, 100, 220)
    width = 4 if is_validated else 2
    draw.rectangle(px, outline=color, width=width)

    if is_validated:
        label = f"{cl.best.template_name} ({cl.best.score:.2f})"
    else:
        label = cand.candidate_id
    draw_label((px[0], max(0, px[1] - 2)), label, color)

img.save(OUT_PATH)
print(f"\nImagem salva em: {OUT_PATH}")
