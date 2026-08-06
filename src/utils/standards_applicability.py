"""
standards_applicability.py
---------------------------
Tópico 2 do prompt_classification.md — importação e normalização da planilha
de aplicabilidade de normas.

Fluxo:
    1. Carrega Normas.xlsx (abas Notes + Parts) em memória
    2. Normaliza nomes de componentes, séries e normas
    3. Expõe função get_applicable_standards() para consulta
    4. Retorna normas aplicáveis baseadas em: componente, série, material

Regras de normalização:
    - Wildcards "All" → aplicável a qualquer valor
    - Listas separadas por vírgula/ponto-e-vírgula → set de valores
    - Case-insensitive matching com similaridade fuzzy
    - Material family → match com Content de normas de categoria Material
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set
from dataclasses import dataclass

import pandas as pd

from src.modeling.part_classification_types import (
    ApplicableStandard,
    StandardsApplicabilityResult,
)

logger = logging.getLogger(__name__)

# Caminho padrão da planilha
DEFAULT_NORMAS_PATH = Path("Normas.xlsx")


# ==============================================================================
# Funções de normalização
# ==============================================================================

def _normalize_text(text: str) -> str:
    """
    Normaliza texto para matching: lowercase, sem pontuação extra, espaços únicos.
    
    Exemplos:
        "Connecting Rod" → "connecting rod"
        "SINTERED  METAL" → "sintered metal"
    """
    if not isinstance(text, str):
        return ""
    # Remove pontuação exceto espaços
    normalized = re.sub(r"[^a-z0-9 ]", "", text.lower())
    # Remove espaços duplicados
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_standard(standard: str) -> str:
    """
    Normaliza código de norma para formato consistente.
    
    Regras:
        - Remove hífens e underscores
        - Adiciona espaço entre prefixo e número
        - Uppercase no prefixo
        - Remove espaços duplicados
    
    Exemplos:
        "TSS002611" → "TSS 002611"
        "TSS-002611" → "TSS 002611"
        "tss 002611" → "TSS 002611"
        "TSS  002611" → "TSS 002611"
    """
    if not isinstance(standard, str):
        return ""
    
    # Remove hífens e underscores
    standard = standard.replace("-", " ").replace("_", " ")
    
    # Regex: captura prefixo (letras) + número
    match = re.match(r"^([A-Za-z]+)\s*(\d+.*)$", standard.strip())
    if match:
        prefix = match.group(1).upper()
        number = match.group(2).strip()
        return f"{prefix} {number}"
    
    # Fallback: uppercase e remove espaços duplicados
    return re.sub(r"\s+", " ", standard.upper().strip())


def _split_list_field(field_value: str) -> Set[str]:
    """
    Separa campo de lista (separado por vírgula ou ponto-e-vírgula).
    
    Exemplos:
        "F, EG, FMF, VEG" → {"F", "EG", "FMF", "VEG"}
        "Crankshaft; Connecting Rod" → {"Crankshaft", "Connecting Rod"}
        "All" → {"all"}
    """
    if not isinstance(field_value, str) or not field_value.strip():
        return set()
    
    # Separa por vírgula ou ponto-e-vírgula
    items = re.split(r"[;,]", field_value)
    # Remove espaços e filtra vazios
    items = [item.strip() for item in items if item.strip()]
    return set(items)


def _similarity_score(a: str, b: str) -> float:
    """
    Calcula similaridade entre duas strings normalizadas (Jaccard sobre tokens).
    
    Retorna valor entre 0.0 (totalmente diferente) e 1.0 (idêntico).
    """
    tokens_a = set(_normalize_text(a).split())
    tokens_b = set(_normalize_text(b).split())
    
    if not tokens_a or not tokens_b:
        return 0.0
    
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    
    return len(intersection) / len(union) if union else 0.0


def _best_match(target: str, candidates: List[str], threshold: float = 0.5) -> Optional[str]:
    """
    Encontra o melhor match fuzzy em uma lista de candidatos.
    
    Args:
        target: String a ser matcheada
        candidates: Lista de candidatos
        threshold: Limiar mínimo de similaridade (0.0-1.0)
    
    Returns:
        Melhor candidato ou None se abaixo do threshold
    """
    best_score = 0.0
    best_candidate = None
    
    for candidate in candidates:
        score = _similarity_score(target, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    if best_score >= threshold:
        return best_candidate
    return None


# ==============================================================================
# Classe principal
# ==============================================================================

@dataclass
class StandardEntry:
    """Uma entrada da aba Notes."""
    standard: str
    content: str
    category: str
    compressor_series: Set[str]  # pode conter "All" ou lista de séries
    applicability: Set[str]  # componentes onde se aplica


class StandardsApplicabilityEngine:
    """
    Motor de consulta de normas aplicáveis baseado em Normas.xlsx.
    
    Carrega as abas Notes e Parts, normaliza os dados e expõe a função
    get_applicable_standards() para consulta.
    """
    
    def __init__(self, normas_path: Path = DEFAULT_NORMAS_PATH):
        self._path = Path(normas_path)
        self._notes: List[StandardEntry] = []
        self._parts_mandatory: dict[str, Set[str]] = {}  # Part → set de normas obrigatórias
        self._known_parts: List[str] = []
        self._loaded = False
    
    def _load(self):
        """Carrega e normaliza as abas Notes e Parts (lazy load)."""
        if self._loaded:
            return
        
        if not self._path.exists():
            raise FileNotFoundError(
                f"Planilha de normas não encontrada: {self._path.resolve()}\n"
                "Verifique se o arquivo Normas.xlsx está na raiz do projeto."
            )
        
        logger.info(f"Carregando planilha de normas: {self._path}")
        
        # Carregar aba Notes
        df_notes = pd.read_excel(self._path, sheet_name="Notes")
        for _, row in df_notes.iterrows():
            standard = _normalize_standard(str(row.get("Standard", "")))
            if not standard:
                continue
            
            content = str(row.get("Content", "")).strip()
            category = str(row.get("Category", "")).strip()
            
            # Compressor_Series pode ser "All" ou lista separada por vírgula
            series_raw = str(row.get("Compressor_Series", ""))
            compressor_series = _split_list_field(series_raw)
            
            # Applicability: lista de componentes
            applicability_raw = str(row.get("Applicability", ""))
            applicability = _split_list_field(applicability_raw)
            
            self._notes.append(StandardEntry(
                standard=standard,
                content=content,
                category=category,
                compressor_series=compressor_series,
                applicability=applicability,
            ))
        
        # Carregar aba Parts
        df_parts = pd.read_excel(self._path, sheet_name="Parts")
        for _, row in df_parts.iterrows():
            part = str(row.get("Part", "")).strip()
            if not part:
                continue
            
            # Mandatory_Standards: separado por ponto-e-vírgula
            mandatory_raw = str(row.get("Mandatory_Standards", ""))
            mandatory = _split_list_field(mandatory_raw)
            # Normalizar códigos de normas
            mandatory_normalized = {_normalize_standard(s) for s in mandatory}
            
            self._parts_mandatory[part] = mandatory_normalized
            self._known_parts.append(part)
        
        self._loaded = True
        logger.info(
            f"✅ Normas.xlsx carregado — {len(self._notes)} normas, "
            f"{len(self._parts_mandatory)} tipos de peça"
        )
    
    def get_applicable_standards(
        self,
        component: str,
        compressor_series: Optional[str] = None,
        material_family: Optional[str] = None,
        match_threshold: float = 0.5,
    ) -> StandardsApplicabilityResult:
        """
        Consulta as normas aplicáveis para um componente.
        
        Args:
            component: Tipo do componente (ex: "Connecting Rod")
            compressor_series: Série do compressor (ex: "F") — opcional
            material_family: Família do material (ex: "sintered_metal") — opcional
            match_threshold: Limiar de similaridade fuzzy (0.0-1.0)
        
        Returns:
            StandardsApplicabilityResult com normas aplicáveis e campos não resolvidos
        
        Lógica:
            1. Fuzzy match do componente na aba Parts → normas obrigatórias
            2. Filtra normas da aba Notes por:
               - Applicability contém o componente
               - Compressor_Series é "All" ou contém a série fornecida
            3. Match de material: normas de categoria Material cujo Content menciona a família
        """
        self._load()
        
        applicable_standards: List[ApplicableStandard] = []
        unresolved_fields: List[str] = []
        
        # === 1. Match do componente na aba Parts ===
        matched_part = _best_match(component, self._known_parts, match_threshold)
        
        if matched_part:
            logger.info(f"✅ Match: '{component}' → '{matched_part}' (Parts)")
            mandatory = self._parts_mandatory.get(matched_part, set())
            
            for std in mandatory:
                applicable_standards.append(ApplicableStandard(
                    standard=std,
                    reason=f"Norma obrigatória para '{matched_part}' (aba Parts)",
                    source="customer_applicability_matrix",
                ))
        else:
            logger.warning(f"⚠️  Componente '{component}' não encontrado na aba Parts")
        
        # === 2. Normas da aba Notes aplicáveis ao componente ===
        for entry in self._notes:
            # Verifica se componente está na lista de Applicability
            component_match = any(
                _similarity_score(component, app) >= match_threshold
                for app in entry.applicability
            )
            
            # Verifica se "All" está na lista de aplicabilidade (aplica-se a tudo)
            is_all_components = "All" in entry.applicability or "all" in {a.lower() for a in entry.applicability}
            
            if not component_match and not is_all_components:
                continue
            
            # Verifica série do compressor
            series_match = True
            if compressor_series:
                # "All" significa que se aplica a qualquer série
                is_all_series = "All" in entry.compressor_series or "all" in {s.lower() for s in entry.compressor_series}
                
                if not is_all_series:
                    # Verifica se a série está na lista
                    series_match = compressor_series in entry.compressor_series
            else:
                # Série não fornecida — só aceita normas "All"
                is_all_series = "All" in entry.compressor_series or "all" in {s.lower() for s in entry.compressor_series}
                if not is_all_series:
                    # Não podemos resolver se esta norma se aplica
                    if "compressor_series" not in unresolved_fields:
                        unresolved_fields.append("compressor_series")
                    series_match = False
            
            if not series_match:
                continue
            
            # Se chegou aqui, a norma é aplicável
            # Evita duplicatas
            if entry.standard not in {s.standard for s in applicable_standards}:
                applicable_standards.append(ApplicableStandard(
                    standard=entry.standard,
                    reason=f"{entry.content} (categoria: {entry.category})",
                    source="component_match",
                ))
        
        # === 3. Match de material (normas de categoria Material) ===
        if material_family:
            material_normalized = _normalize_text(material_family)
            
            for entry in self._notes:
                if entry.category.lower() != "material":
                    continue
                
                content_normalized = _normalize_text(entry.content)
                
                # Verifica se o material_family aparece no Content da norma
                if material_normalized in content_normalized or _similarity_score(material_family, entry.content) >= 0.4:
                    # Evita duplicatas
                    if entry.standard not in {s.standard for s in applicable_standards}:
                        applicable_standards.append(ApplicableStandard(
                            standard=entry.standard,
                            reason=f"Material: {entry.content}",
                            source="material_match",
                        ))
        
        return StandardsApplicabilityResult(
            component=component,
            compressor_series=compressor_series,
            material_family=material_family,
            applicable_standards=applicable_standards,
            unresolved_fields=unresolved_fields,
        )
    
    def list_known_parts(self) -> List[str]:
        """Retorna a lista de componentes conhecidos na aba Parts."""
        self._load()
        return self._known_parts.copy()


# ==============================================================================
# Instância global (lazy loaded)
# ==============================================================================

_engine: Optional[StandardsApplicabilityEngine] = None


def get_applicable_standards(
    component: str,
    compressor_series: Optional[str] = None,
    material_family: Optional[str] = None,
    normas_path: Path = DEFAULT_NORMAS_PATH,
) -> StandardsApplicabilityResult:
    """
    Função de conveniência para consultar normas aplicáveis.
    
    Usa instância global do engine (lazy loaded).
    
    Args:
        component: Tipo do componente (ex: "Connecting Rod")
        compressor_series: Série do compressor (ex: "F") — opcional
        material_family: Família do material (ex: "sintered_metal") — opcional
        normas_path: Caminho da planilha (padrão: Normas.xlsx na raiz)
    
    Returns:
        StandardsApplicabilityResult
    """
    global _engine
    if _engine is None:
        _engine = StandardsApplicabilityEngine(normas_path)
    
    return _engine.get_applicable_standards(
        component=component,
        compressor_series=compressor_series,
        material_family=material_family,
    )
