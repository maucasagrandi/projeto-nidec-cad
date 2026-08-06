"""
validate_group2.py
------------------
Validação do Grupo 2 (Tópicos 3 + 4):
- LLM classificação enriquecida (com evidências + confiança)
- Normalização de normas citadas

Testa com 2 PDFs de cad_docs_examples/:
- 19308765_rev00.pdf
- 19308765_rev01.pdf
"""

import json
import logging
from pathlib import Path

from prompts import classificacao_enriquecida_prompt
from src.modeling.llm_models import classify_cad_enriched, extract_text_from_pdf
from src.modeling.part_classification_types import CitedStandard
from src.utils.standards_applicability import normalize_standard, extract_note_number

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

def print_separator(title: str):
    """Imprime separador visual."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def validate_pdf(pdf_path: Path, test_name: str):
    """Valida um PDF com o pipeline completo do Grupo 2."""
    print_separator(test_name)
    
    print(f"📄 PDF: {pdf_path.name}")
    print()
    
    if not pdf_path.exists():
        print(f"❌ Arquivo não encontrado: {pdf_path}")
        return
    
    # =========================================================================
    # Etapa 1: Extração de texto
    # =========================================================================
    print("📝 Etapa 1: Extraindo texto do PDF...")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    texto = extract_text_from_pdf(pdf_bytes, page_index=0)
    print(f"   Texto extraído: {len(texto)} caracteres")
    print(f"   Primeiros 300 chars: {texto[:300]}...")
    print()
    
    # =========================================================================
    # Etapa 2: Classificação enriquecida (Tópico 3)
    # =========================================================================
    print("🤖 Etapa 2: Classificação enriquecida com LLM...")
    
    result, metadata = classify_cad_enriched(
        texto_extraido=texto,
        system_prompt=classificacao_enriquecida_prompt,
    )
    
    print(f"   ✅ Classificação concluída")
    print(f"   Tokens: {metadata.total_tokens} (input: {metadata.prompt_tokens}, output: {metadata.completion_tokens})")
    print(f"   Latência: {metadata.latency_ms:.0f}ms")
    print()
    
    # =========================================================================
    # Etapa 3: Exibir resultados estruturados
    # =========================================================================
    print("📊 Resultados:")
    print()
    
    print(f"   🏷️  DOCUMENT TYPE:")
    print(f"      Value: {result.document_type.value}")
    print(f"      Evidence: {result.document_type.evidence}")
    print(f"      Confidence: {result.document_type.confidence:.2f}")
    print()
    
    print(f"   🔩 COMPONENT:")
    print(f"      Value: {result.component.value}")
    print(f"      Evidence: {result.component.evidence}")
    print(f"      Confidence: {result.component.confidence:.2f}")
    print()
    
    print(f"   🧪 MATERIAL FAMILY:")
    print(f"      Value: {result.material_family.value}")
    print(f"      Evidence: {result.material_family.evidence}")
    print(f"      Confidence: {result.material_family.confidence:.2f}")
    print()
    
    print(f"   📦 COMPRESSOR SERIES:")
    print(f"      Value: {result.compressor_series.value}")
    print(f"      Evidence: {result.compressor_series.evidence}")
    print(f"      Confidence: {result.compressor_series.confidence:.2f}")
    print()
    
    print(f"   📋 CITED STANDARDS (raw): {len(result.cited_standards)}")
    for i, std in enumerate(result.cited_standards, 1):
        print(f"      {i}. {std.standard}")
        print(f"         Evidence: {std.evidence[:80]}...")
    print()
    
    # =========================================================================
    # Etapa 4: Normalização de normas (Tópico 4)
    # =========================================================================
    print("🔄 Etapa 3: Normalizando normas citadas (Tópico 4)...")
    
    cited_standards_normalized: list[CitedStandard] = []
    
    for std_raw in result.cited_standards:
        # Normalizar código
        standard_normalized = normalize_standard(std_raw.standard)
        
        # Extrair número da nota
        note_number = extract_note_number(std_raw.evidence)
        
        cited_std = CitedStandard(
            standard=standard_normalized,
            standard_raw=std_raw.standard,
            note_number=note_number,
            source_text=std_raw.evidence,
        )
        cited_standards_normalized.append(cited_std)
    
    print(f"   ✅ {len(cited_standards_normalized)} normas normalizadas")
    print()
    
    print("   📋 CITED STANDARDS (normalized):")
    for i, std in enumerate(cited_standards_normalized, 1):
        note_str = f"Note {std.note_number}" if std.note_number else "—"
        print(f"      {i}. {std.standard_raw:20s} → {std.standard:20s} ({note_str})")
        print(f"         Source: {std.source_text[:70]}...")
    print()
    
    # =========================================================================
    # Etapa 5: Exportar JSON consolidado
    # =========================================================================
    output_data = {
        "pdf_file": pdf_path.name,
        "classification": {
            "document_type": result.document_type.model_dump(),
            "component": result.component.model_dump(),
            "material_family": result.material_family.model_dump(),
            "compressor_series": result.compressor_series.model_dump(),
        },
        "cited_standards_normalized": [std.model_dump() for std in cited_standards_normalized],
        "metadata": {
            "total_tokens": metadata.total_tokens,
            "prompt_tokens": metadata.prompt_tokens,
            "completion_tokens": metadata.completion_tokens,
            "latency_ms": metadata.latency_ms,
            "model": metadata.model,
            "timestamp": metadata.timestamp,
        }
    }
    
    output_file = Path(f"validation_group2_{pdf_path.stem}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON exportado: {output_file}")
    print()


def main():
    """Executa validação do Grupo 2 com 2 PDFs."""
    
    print_separator("VALIDAÇÃO GRUPO 2 — TÓPICOS 3 + 4")
    print("LLM classificação enriquecida + normalização de normas")
    print()
    
    cad_examples = Path("cad_docs_examples")
    
    if not cad_examples.exists():
        print(f"❌ Erro: pasta {cad_examples} não encontrada")
        return
    
    # =========================================================================
    # Teste 1: 19308765_rev00.pdf
    # =========================================================================
    validate_pdf(
        pdf_path=cad_examples / "19308765_rev00.pdf",
        test_name="Teste 1 - 19308765_rev00.pdf"
    )
    
    # =========================================================================
    # Teste 2: 19308765_rev01.pdf
    # =========================================================================
    validate_pdf(
        pdf_path=cad_examples / "19308765_rev01.pdf",
        test_name="Teste 2 - 19308765_rev01.pdf"
    )
    
    # =========================================================================
    # Sumário final
    # =========================================================================
    print_separator("VALIDAÇÃO CONCLUÍDA")
    print("✅ 2 testes executados")
    print("✅ Arquivos JSON gerados na raiz do projeto")
    print()
    print("📋 Verificações necessárias:")
    print("   1. Conferir se os campos value/evidence/confidence fazem sentido")
    print("   2. Verificar se série foi null quando não havia evidência")
    print("   3. Verificar se normas foram normalizadas corretamente")
    print("   4. Conferir se números de notas foram extraídos quando presentes")
    print()
    print("📋 Próximo passo:")
    print("   Implementar Grupo 3 (Tópico 5): Comparação determinística de normas")
    print()


if __name__ == "__main__":
    main()
