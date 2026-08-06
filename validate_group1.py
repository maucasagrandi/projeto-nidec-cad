"""
validate_group1.py
------------------
Validação do Grupo 1 (Tópicos 1 + 2):
- Contratos de dados
- Importação da planilha de aplicabilidade

Testa a função get_applicable_standards() com 5 componentes diferentes
e verifica se as normas retornadas fazem sentido.
"""

import json
import logging
from pathlib import Path

from src.utils.standards_applicability import (
    get_applicable_standards,
    StandardsApplicabilityEngine,
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


def validate_test_case(
    test_name: str,
    component: str,
    compressor_series: str = None,
    material_family: str = None,
):
    """Executa um caso de teste e imprime o resultado."""
    print_separator(test_name)
    
    print(f"📋 Entrada:")
    print(f"  - Componente: {component}")
    print(f"  - Série: {compressor_series or 'Não fornecida'}")
    print(f"  - Material: {material_family or 'Não fornecido'}")
    print()
    
    result = get_applicable_standards(
        component=component,
        compressor_series=compressor_series,
        material_family=material_family,
    )
    
    print(f"✅ Normas aplicáveis encontradas: {len(result.applicable_standards)}")
    print()
    
    if result.unresolved_fields:
        print(f"⚠️  Campos não resolvidos: {', '.join(result.unresolved_fields)}")
        print()
    
    # Agrupar por source
    by_source = {}
    for std in result.applicable_standards:
        if std.source not in by_source:
            by_source[std.source] = []
        by_source[std.source].append(std)
    
    for source, standards in by_source.items():
        print(f"📌 Fonte: {source}")
        for std in standards:
            print(f"   {std.standard:20s} | {std.reason}")
        print()
    
    # Exportar JSON
    output_file = Path(f"validation_output_{test_name.lower().replace(' ', '_')}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON exportado: {output_file}")
    print()


def main():
    """Executa validação com 5 componentes diferentes."""
    
    print_separator("VALIDAÇÃO GRUPO 1 — TÓPICOS 1 + 2")
    print("Testando importação da planilha e consulta de normas aplicáveis")
    print()
    
    # Verificar se Normas.xlsx existe
    normas_path = Path("Normas.xlsx")
    if not normas_path.exists():
        print(f"❌ Erro: {normas_path.resolve()} não encontrado")
        print("   Execute este script na raiz do projeto onde está o arquivo Normas.xlsx")
        return
    
    print(f"✅ Planilha encontrada: {normas_path.resolve()}")
    print()
    
    # Carregar engine para listar componentes conhecidos
    engine = StandardsApplicabilityEngine(normas_path)
    known_parts = engine.list_known_parts()
    
    print(f"📦 Componentes conhecidos na planilha ({len(known_parts)}):")
    for part in sorted(known_parts):
        print(f"   - {part}")
    print()
    
    # =========================================================================
    # Teste 1: Connecting Rod com série F e material sintered_metal
    # =========================================================================
    validate_test_case(
        test_name="Teste 1 - Connecting Rod (série F, sintered_metal)",
        component="Connecting Rod",
        compressor_series="F",
        material_family="sintered_metal",
    )
    
    # =========================================================================
    # Teste 2: Crankshaft com série EG (sem material)
    # =========================================================================
    validate_test_case(
        test_name="Teste 2 - Crankshaft (série EG)",
        component="Crankshaft",
        compressor_series="EG",
        material_family=None,
    )
    
    # =========================================================================
    # Teste 3: Piston sem série (deve marcar como INCONCLUSIVE)
    # =========================================================================
    validate_test_case(
        test_name="Teste 3 - Piston (sem série)",
        component="Piston",
        compressor_series=None,
        material_family=None,
    )
    
    # =========================================================================
    # Teste 4: Baseplates com série EM e steel sheet
    # =========================================================================
    validate_test_case(
        test_name="Teste 4 - Baseplates (série EM, steel sheet)",
        component="Baseplates",
        compressor_series="EM",
        material_family="steel sheet",
    )
    
    # =========================================================================
    # Teste 5: Componente desconhecido (fuzzy match ou falha)
    # =========================================================================
    validate_test_case(
        test_name="Teste 5 - Connecting rod raw (fuzzy match)",
        component="connecting rod raw",
        compressor_series="F",
        material_family="sintered metal",
    )
    
    # =========================================================================
    # Sumário final
    # =========================================================================
    print_separator("VALIDAÇÃO CONCLUÍDA")
    print("✅ 5 testes executados")
    print("✅ Arquivos JSON gerados na raiz do projeto")
    print()
    print("📋 Próximos passos:")
    print("   1. Revisar os JSONs gerados")
    print("   2. Verificar se as normas retornadas fazem sentido")
    print("   3. Conferir se os campos unresolved_fields estão corretos")
    print("   4. Passar para o Grupo 2 (Tópicos 3 + 4)")
    print()


if __name__ == "__main__":
    main()
