"""
validate_group3.py
------------------
Validação do Grupo 3 (Tópico 5):
- Comparação determinística de normas (matching, missing, unexpected)

Pipeline completo end-to-end:
1. Grupo 1: Consulta planilha → normas aplicáveis
2. Grupo 2: LLM classifica + normaliza → normas citadas
3. Grupo 3: Comparação determinística → diff de normas

Este é o MARCO 1 DA FASE 1 — primeira validação completa do pipeline de normas.
"""

import json
import logging
from pathlib import Path

from prompts import classificacao_enriquecida_prompt
from src.modeling.llm_models import classify_cad_enriched, extract_text_from_pdf
from src.modeling.part_classification_types import CitedStandard
from src.utils.standards_applicability import (
    get_applicable_standards,
    normalize_standard,
    extract_note_number,
    compare_standards,
)

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


def validate_end_to_end(pdf_path: Path, test_name: str):
    """
    Executa o pipeline completo end-to-end:
    Grupo 1 + Grupo 2 + Grupo 3
    """
    print_separator(test_name)
    
    print(f"📄 PDF: {pdf_path.name}")
    print()
    
    if not pdf_path.exists():
        print(f"❌ Arquivo não encontrado: {pdf_path}")
        return
    
    # =========================================================================
    # FASE 1: Extração de texto do PDF
    # =========================================================================
    print("📝 Fase 1: Extraindo texto do PDF...")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    texto = extract_text_from_pdf(pdf_bytes, page_index=0)
    print(f"   ✅ Texto extraído: {len(texto)} caracteres")
    print()
    
    # =========================================================================
    # FASE 2: Classificação enriquecida com LLM (Grupo 2 - Tópico 3)
    # =========================================================================
    print("🤖 Fase 2: Classificação enriquecida com LLM (Tópico 3)...")
    
    classification_result, classification_metadata = classify_cad_enriched(
        texto_extraido=texto,
        system_prompt=classificacao_enriquecida_prompt,
    )
    
    print(f"   ✅ Classificação concluída")
    print(f"   Component: {classification_result.component.value} (conf={classification_result.component.confidence:.2f})")
    print(f"   Material: {classification_result.material_family.value} (conf={classification_result.material_family.confidence:.2f})")
    print(f"   Series: {classification_result.compressor_series.value} (conf={classification_result.compressor_series.confidence:.2f})")
    print(f"   Standards cited: {len(classification_result.cited_standards)}")
    print()
    
    # =========================================================================
    # FASE 3: Normalização de normas (Grupo 2 - Tópico 4)
    # =========================================================================
    print("🔄 Fase 3: Normalizando normas citadas (Tópico 4)...")
    
    cited_standards_normalized: list[CitedStandard] = []
    
    for std_raw in classification_result.cited_standards:
        standard_normalized = normalize_standard(std_raw.standard)
        note_number = extract_note_number(std_raw.evidence)
        
        cited_std = CitedStandard(
            standard=standard_normalized,
            standard_raw=std_raw.standard,
            note_number=note_number,
            source_text=std_raw.evidence,
        )
        cited_standards_normalized.append(cited_std)
    
    print(f"   ✅ {len(cited_standards_normalized)} normas normalizadas")
    for std in cited_standards_normalized:
        print(f"      {std.standard_raw:20s} → {std.standard}")
    print()
    
    # =========================================================================
    # FASE 4: Consulta de normas aplicáveis (Grupo 1 - Tópico 2)
    # =========================================================================
    print("📊 Fase 4: Consultando normas aplicáveis (Tópico 2)...")
    
    applicability_result = get_applicable_standards(
        component=classification_result.component.value or "Unknown",
        compressor_series=classification_result.compressor_series.value,
        material_family=classification_result.material_family.value,
    )
    
    print(f"   ✅ {len(applicability_result.applicable_standards)} normas aplicáveis")
    if applicability_result.unresolved_fields:
        print(f"   ⚠️  Campos não resolvidos: {', '.join(applicability_result.unresolved_fields)}")
    
    for std in applicability_result.applicable_standards:
        print(f"      {std.standard:20s} ({std.source})")
    print()
    
    # =========================================================================
    # FASE 5: Comparação determinística (Grupo 3 - Tópico 5)
    # =========================================================================
    print("⚖️  Fase 5: Comparação determinística de normas (Tópico 5)...")
    
    # Extrair lista de strings das normas aplicáveis
    applicable_standards_list = [std.standard for std in applicability_result.applicable_standards]
    
    # Extrair lista de strings das normas citadas (normalizadas)
    cited_standards_list = [std.standard for std in cited_standards_normalized]
    
    # Comparação determinística
    comparison_result = compare_standards(
        applicable_standards=applicable_standards_list,
        cited_standards=cited_standards_list,
        unresolved_fields=applicability_result.unresolved_fields,
    )
    
    print(f"   ✅ Comparação concluída")
    print(f"   Status: {comparison_result.applicability_status}")
    print()
    
    # =========================================================================
    # FASE 6: Exibir resultados estruturados
    # =========================================================================
    print("📋 Resultados da Comparação:")
    print()
    
    print(f"   📌 NORMAS ESPERADAS: {len(comparison_result.expected)}")
    for std in comparison_result.expected:
        print(f"      {std}")
    print()
    
    print(f"   📌 NORMAS CITADAS: {len(comparison_result.cited)}")
    for std in comparison_result.cited:
        print(f"      {std}")
    print()
    
    print(f"   ✅ NORMAS PRESENTES (matching): {len(comparison_result.matching)}")
    for std in comparison_result.matching:
        print(f"      {std}")
    print()
    
    print(f"   ⚠️  NORMAS FALTANTES (missing): {len(comparison_result.missing)}")
    if comparison_result.missing:
        for std in comparison_result.missing:
            # Buscar detalhes da norma faltante
            detail = next(
                (s for s in applicability_result.applicable_standards if s.standard == std),
                None
            )
            if detail:
                print(f"      {std:20s} | {detail.reason}")
            else:
                print(f"      {std}")
    else:
        print(f"      (Nenhuma norma obrigatória faltante)")
    print()
    
    print(f"   ❓ NORMAS EXTRAS (unexpected): {len(comparison_result.unexpected)}")
    if comparison_result.unexpected:
        for std in comparison_result.unexpected:
            print(f"      {std}")
    else:
        print(f"      (Sem normas extras não esperadas)")
    print()
    
    # Percentual de conformidade
    if comparison_result.expected:
        conformity_pct = len(comparison_result.matching) / len(comparison_result.expected)
        print(f"   📊 CONFORMIDADE: {len(comparison_result.matching)}/{len(comparison_result.expected)} ({conformity_pct:.1%})")
        print()
    
    # =========================================================================
    # FASE 7: Exportar JSON consolidado
    # =========================================================================
    output_data = {
        "pdf_file": pdf_path.name,
        "classification": {
            "document_type": classification_result.document_type.model_dump(),
            "component": classification_result.component.model_dump(),
            "material_family": classification_result.material_family.model_dump(),
            "compressor_series": classification_result.compressor_series.model_dump(),
        },
        "cited_standards": [std.model_dump() for std in cited_standards_normalized],
        "applicable_standards": [std.model_dump() for std in applicability_result.applicable_standards],
        "unresolved_fields": applicability_result.unresolved_fields,
        "comparison": comparison_result.model_dump(),
        "metadata": {
            "total_tokens": classification_metadata.total_tokens,
            "prompt_tokens": classification_metadata.prompt_tokens,
            "completion_tokens": classification_metadata.completion_tokens,
            "latency_ms": classification_metadata.latency_ms,
            "model": classification_metadata.model,
            "timestamp": classification_metadata.timestamp,
        }
    }
    
    output_file = Path(f"validation_group3_{pdf_path.stem}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON consolidado exportado: {output_file}")
    print()


def main():
    """Executa validação completa da Fase 1 (Grupos 1 + 2 + 3)."""
    
    print_separator("VALIDAÇÃO GRUPO 3 — TÓPICO 5 (MARCO 1 DA FASE 1)")
    print("Pipeline completo end-to-end:")
    print("  Grupo 1: Normas aplicáveis (planilha)")
    print("  Grupo 2: Classificação LLM + normalização")
    print("  Grupo 3: Comparação determinística")
    print()
    
    cad_examples = Path("cad_docs_examples")
    
    if not cad_examples.exists():
        print(f"❌ Erro: pasta {cad_examples} não encontrada")
        return
    
    # =========================================================================
    # Teste 1: 19308765_rev00.pdf
    # =========================================================================
    validate_end_to_end(
        pdf_path=cad_examples / "19308765_rev00.pdf",
        test_name="Teste 1 - 19308765_rev00.pdf (Pipeline Completo)"
    )
    
    # =========================================================================
    # Teste 2: 19308765_rev01.pdf
    # =========================================================================
    validate_end_to_end(
        pdf_path=cad_examples / "19308765_rev01.pdf",
        test_name="Teste 2 - 19308765_rev01.pdf (Pipeline Completo)"
    )
    
    # =========================================================================
    # Sumário final
    # =========================================================================
    print_separator("✅ MARCO 1 DA FASE 1 CONCLUÍDO")
    print("Pipeline de normas validado end-to-end:")
    print()
    print("✅ Grupo 1: Normas aplicáveis (planilha + fuzzy match)")
    print("✅ Grupo 2: Classificação LLM enriquecida + normalização")
    print("✅ Grupo 3: Comparação determinística (matching/missing/unexpected)")
    print()
    print("📋 Arquivos gerados:")
    print("   - validation_group3_19308765_rev00.json")
    print("   - validation_group3_19308765_rev01.json")
    print()
    print("📋 Verificações necessárias:")
    print("   1. Conferir se normas faltantes fazem sentido")
    print("   2. Verificar se normas extras são justificáveis")
    print("   3. Validar percentual de conformidade")
    print("   4. Conferir se unresolved_fields está correto")
    print()
    print("🎯 Próximos passos:")
    print("   FASE 1 COMPLETA — pronto para:")
    print("   - Integração com frontend (pages/classification.py)")
    print("   - OU seguir para Fase 2: Catálogo GD&T (Tópico 6)")
    print()


if __name__ == "__main__":
    main()
