"""
part_classification_types.py
-----------------------------
Contratos de dados (Pydantic models) para o pipeline de Part Classification.

Tópico 1 do prompt_classification.md — estruturas de entrada e saída para:
- Classificação de componentes CAD
- Aplicabilidade de normas
- Observações de GD&T (geometrical tolerancing)
- Definições de datum
- Findings de conformidade

Todas as estruturas são validadas via Pydantic para garantir tipagem forte
e serialização JSON consistente.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ==============================================================================
# Classificação de componentes CAD
# ==============================================================================

class FieldWithEvidence(BaseModel):
    """Campo de classificação com evidência textual e confiança."""
    value: Optional[str] = Field(
        default=None,
        description="Valor extraído (None se não encontrado)"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Trecho do texto que sustenta esta classificação"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Nível de confiança (0.0 a 1.0)"
    )


class CadClassification(BaseModel):
    """
    Classificação completa de um componente CAD extraída via LLM.
    
    Todos os campos opcionais devem ter evidence=None e confidence=0.0
    quando não encontrados no texto do CAD.
    """
    document_type: FieldWithEvidence = Field(
        description="Tipo do documento (ex: product_drawing, assembly_drawing)"
    )
    component: FieldWithEvidence = Field(
        description="Tipo do componente (ex: Connecting Rod, Crankshaft)"
    )
    material_family: FieldWithEvidence = Field(
        description="Família do material (ex: sintered_metal, gray_cast_iron)"
    )
    compressor_series: FieldWithEvidence = Field(
        description="Série do compressor (ex: F, EG, EM) — só quando houver evidência"
    )
    cited_standards_raw: List[str] = Field(
        default_factory=list,
        description="Lista bruta de normas citadas antes da normalização"
    )


# ==============================================================================
# Aplicabilidade de normas
# ==============================================================================

class ApplicableStandard(BaseModel):
    """Uma norma aplicável identificada pela consulta à planilha."""
    standard: str = Field(
        description="Código da norma (ex: TSS 002611)"
    )
    reason: str = Field(
        description="Razão pela qual esta norma é aplicável"
    )
    source: Literal["customer_applicability_matrix", "material_match", "component_match"] = Field(
        description="Fonte da aplicabilidade"
    )


class StandardsApplicabilityResult(BaseModel):
    """Resultado da consulta de normas aplicáveis."""
    component: str = Field(
        description="Componente consultado"
    )
    compressor_series: Optional[str] = Field(
        default=None,
        description="Série consultada (None se não fornecida)"
    )
    material_family: Optional[str] = Field(
        default=None,
        description="Família do material consultada"
    )
    applicable_standards: List[ApplicableStandard] = Field(
        default_factory=list,
        description="Lista de normas aplicáveis encontradas"
    )
    unresolved_fields: List[str] = Field(
        default_factory=list,
        description="Campos que não puderam ser resolvidos (ex: ['compressor_series'])"
    )


class CitedStandard(BaseModel):
    """Uma norma citada no CAD após normalização."""
    standard: str = Field(
        description="Código normalizado da norma (ex: TSS 002611)"
    )
    standard_raw: str = Field(
        description="Código original antes da normalização (ex: TSS002611)"
    )
    note_number: Optional[int] = Field(
        default=None,
        description="Número da nota onde a norma foi citada (se aplicável)"
    )
    source_text: str = Field(
        description="Trecho do texto que menciona esta norma"
    )


class StandardsComparisonResult(BaseModel):
    """Resultado da comparação determinística entre normas esperadas e citadas."""
    expected: List[str] = Field(
        description="Normas esperadas (aplicáveis) para este componente"
    )
    cited: List[str] = Field(
        description="Normas citadas no CAD"
    )
    matching: List[str] = Field(
        description="Normas presentes e esperadas"
    )
    missing: List[str] = Field(
        description="Normas esperadas mas ausentes no CAD"
    )
    unexpected: List[str] = Field(
        description="Normas presentes mas não esperadas"
    )
    applicability_status: Literal["RESOLVED", "INCONCLUSIVE"] = Field(
        description="Status da resolução de aplicabilidade"
    )


# ==============================================================================
# GD&T (Geometrical Dimensioning and Tolerancing)
# ==============================================================================

class GdtObservation(BaseModel):
    """
    Observação de um quadro de tolerância geométrica detectado no CAD.
    
    Produzida após detecção do quadro (Tópico 7), classificação visual do símbolo
    (Tópico 9) e extração de tolerância/datums (Tópico 10).
    """
    gdt_id: str = Field(
        description="ID único do quadro GD&T (ex: GDT-001)"
    )
    characteristic: str = Field(
        description="Característica geométrica (ex: parallelism, flatness)"
    )
    tolerance_raw: str = Field(
        description="Valor bruto da tolerância extraído (ex: 0,03)"
    )
    referenced_datums: List[str] = Field(
        default_factory=list,
        description="Lista de datums referenciados no quadro (ex: ['A', 'B'])"
    )
    page: int = Field(
        description="Número da página (1-indexed)"
    )
    quadrant: Optional[str] = Field(
        default=None,
        description="Quadrante calculado (ex: 7D)"
    )
    bbox: List[float] = Field(
        description="Bounding box em pontos PDF [x0, y0, x1, y1]"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança da classificação visual do símbolo"
    )


class DatumDefinition(BaseModel):
    """
    Definição de um datum encontrado no desenho.
    
    Detectado via análise visual (Tópico 11) — não é apenas uma letra,
    mas o indicador completo (caixa + triângulo).
    """
    datum: str = Field(
        description="Letra do datum (ex: A, B, C)"
    )
    indicator_type: Literal["datum_feature_indicator", "datum_target_indicator", "unknown"] = Field(
        description="Tipo do indicador visual"
    )
    page: int = Field(
        description="Número da página (1-indexed)"
    )
    quadrant: Optional[str] = Field(
        default=None,
        description="Quadrante calculado (ex: 5C)"
    )
    bbox: List[float] = Field(
        description="Bounding box em pontos PDF [x0, y0, x1, y1]"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança da detecção visual"
    )


# ==============================================================================
# Findings de conformidade
# ==============================================================================

class ComplianceFinding(BaseModel):
    """
    Resultado de uma verificação de conformidade (Tópico 12).
    
    Gerado pelo motor determinístico após cruzar GdtObservation com
    DatumDefinition e as regras do catálogo GD&T.
    """
    finding_id: str = Field(
        description="ID único do finding (ex: F-001)"
    )
    check_type: Literal[
        "datum_required",
        "datum_referenced_exists",
        "standard_applicable_present",
        "standard_applicable_missing",
        "standard_unexpected"
    ] = Field(
        description="Tipo de verificação executada"
    )
    status: Literal["CONFORMING", "NON_CONFORMING", "INCONCLUSIVE", "REQUIRES_HUMAN_REVIEW"] = Field(
        description="Status da conformidade"
    )
    severity: Literal["INFO", "WARNING", "ERROR"] = Field(
        description="Severidade (INFO: conforme, WARNING: observação, ERROR: não conforme)"
    )
    gdt_id: Optional[str] = Field(
        default=None,
        description="ID do quadro GD&T relacionado (se aplicável)"
    )
    standard: Optional[str] = Field(
        default=None,
        description="Código da norma relacionada (se aplicável)"
    )
    quadrants: List[str] = Field(
        default_factory=list,
        description="Lista de quadrantes relacionados (ex: ['7D', '5C'])"
    )
    reason: str = Field(
        description="Explicação técnica do resultado"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Ação recomendada (None se status=CONFORMING)"
    )


# ==============================================================================
# Saída final consolidada
# ==============================================================================

class PartClassificationResult(BaseModel):
    """
    Resultado completo do pipeline de Part Classification.
    
    Consolida todos os artefatos das 5 fases (Tópico 15).
    """
    # Classificação
    classification: CadClassification = Field(
        description="Classificação do componente"
    )
    
    # Normas
    cited_standards: List[CitedStandard] = Field(
        default_factory=list,
        description="Normas citadas no CAD (normalizadas)"
    )
    applicable_standards: List[ApplicableStandard] = Field(
        default_factory=list,
        description="Normas aplicáveis"
    )
    standards_comparison: StandardsComparisonResult = Field(
        description="Comparação normas esperadas × citadas"
    )
    
    # GD&T
    gdt_observations: List[GdtObservation] = Field(
        default_factory=list,
        description="Quadros GD&T detectados"
    )
    datum_definitions: List[DatumDefinition] = Field(
        default_factory=list,
        description="Datums definidos no desenho"
    )
    
    # Conformidade
    findings: List[ComplianceFinding] = Field(
        default_factory=list,
        description="Resultados das verificações de conformidade"
    )
    
    # Metadados
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confiança geral do resultado"
    )
    processing_metadata: dict = Field(
        default_factory=dict,
        description="Metadados de processamento (tokens, latência, etc)"
    )
