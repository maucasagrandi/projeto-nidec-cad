"""
Script para executar classificação em batch nos PDFs de normas
e preencher a planilha de resultados com todas as justificativas.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

from prompts import classificacao_e_normas_prompt, normas_faltantes_prompt
from src.modeling.llm_models import (
    classify_and_extract_norms,
    infer_missing_norms,
    extract_text_from_pdf,
)

# ==============================================================================
# Configuração
# ==============================================================================

PDF_FOLDER = "CAD_Review_Test_Battery_V1/3. Normas"
PLANILHA_PATH = "CAD_Review_Test_Battery_V1/V0 Normas Resultados.xlsx"

# ==============================================================================
# Funções Auxiliares
# ==============================================================================

def get_pdf_files(folder_path):
    """Retorna lista de arquivos PDF ordenada"""
    path = Path(folder_path)
    pdfs = sorted(path.glob("*.pdf"))
    return pdfs


def normalize_normas_list(normas_list):
    """Converte lista de normas para string separada por vírgula"""
    if not normas_list:
        return ""
    return "; ".join(normas_list)


def normalize_justificativas_list(justificativas_list):
    """Converte lista de justificativas para string separada por quebra de linha"""
    if not justificativas_list:
        return ""
    return "\n".join(justificativas_list)


def process_single_pdf(pdf_path):
    """
    Processa um único PDF e retorna os resultados da classificação + normas sugeridas.
    
    Retorna dict com todos os campos para preencher a planilha.
    """
    
    try:
        print(f"  Processando: {pdf_path.name}...", end=" ", flush=True)
        
        # Ler PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        # Extrair texto
        texto_notas = extract_text_from_pdf(pdf_bytes, page_index=0)
        
        # ETAPA 1: Executar classificação + normas
        prompt_completo = classificacao_e_normas_prompt.replace("{{texto_extraido}}", texto_notas)
        
        resultado, metadata_classif = classify_and_extract_norms(
            texto_notas=texto_notas,
            system_prompt=prompt_completo,
        )
        
        # ETAPA 2: Inferir normas faltantes
        inferencia_prompt = normas_faltantes_prompt.format(
            classificacao=resultado.classificacao,
            normas_atuais=", ".join(resultado.lista_normas) if resultado.lista_normas else "Nenhuma"
        )
        
        inferencia_result, metadata_inference = infer_missing_norms(
            classificacao=resultado.classificacao,
            lista_normas_atuais=resultado.lista_normas,
            system_prompt=inferencia_prompt,
        )
        
        # Calcular tokens totais
        total_input_tokens = metadata_classif.prompt_tokens + metadata_inference.prompt_tokens
        total_output_tokens = metadata_classif.completion_tokens + metadata_inference.completion_tokens
        total_latency = metadata_classif.latency_ms + metadata_inference.latency_ms
        
        # Formatar saída
        output = {
            "Classificação_LLM": resultado.classificacao,
            "Justificativa_Classificação": resultado.justificativa_classificacao,
            "Normas_LLM": normalize_normas_list(resultado.lista_normas),
            "Justificativas_Normas": normalize_justificativas_list(resultado.justificativas_normas),
            "Normas_Sugeridas_LLM": normalize_normas_list(inferencia_result.normas_sugeridas),
            "Reasoning_Sugeridas": inferencia_result.reasoning,
            "Input_Tokens": total_input_tokens,
            "Output_Tokens": total_output_tokens,
            "Latência": round(total_latency, 2),
            "Erro": None,
        }
        
        print("[OK]")
        return output
        
    except Exception as e:
        print(f"[ERRO] {str(e)}")
        return {
            "Classificação_LLM": None,
            "Justificativa_Classificação": None,
            "Normas_LLM": None,
            "Justificativas_Normas": None,
            "Normas_Sugeridas_LLM": None,
            "Reasoning_Sugeridas": None,
            "Input_Tokens": None,
            "Output_Tokens": None,
            "Latência": None,
            "Erro": str(e),
        }


def main():
    """Executa o pipeline completo"""
    
    print("=" * 80)
    print("[*] BATCH CLASSIFICATION - Normas")
    print("=" * 80)
    
    # 1. Carregar planilha existente
    print("\n[*] Carregando planilha existente...")
    try:
        df = pd.read_excel(PLANILHA_PATH)
        print(f"   [OK] {len(df)} registros encontrados")
    except FileNotFoundError:
        print(f"   [ERRO] Arquivo não encontrado: {PLANILHA_PATH}")
        return
    
    # 2. Obter lista de PDFs
    print(f"\n[*] Buscando PDFs em: {PDF_FOLDER}")
    pdf_files = get_pdf_files(PDF_FOLDER)
    print(f"   [OK] {len(pdf_files)} PDFs encontrados")
    
    # 3. Garantir que as colunas existem
    required_columns = [
        "Classificação_LLM",
        "Justificativa_Classificação",
        "Normas_LLM",
        "Justificativas_Normas",
        "Normas_Sugeridas_LLM",
        "Reasoning_Sugeridas",
        "Input_Tokens",
        "Output_Tokens",
        "Latência",
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # 4. Processar cada PDF
    print(f"\n[*] Processando {len(pdf_files)} PDFs:\n")
    
    for i, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.name
        
        # Encontrar a linha correspondente na planilha
        row_idx = df[df["CAD"] == pdf_name].index
        
        if len(row_idx) == 0:
            print(f"  [!] PDF {pdf_name} não encontrado na planilha")
            continue
        
        row_idx = row_idx[0]
        
        # Processar PDF
        resultado = process_single_pdf(pdf_path)
        
        # Preencher linha da planilha
        for col, value in resultado.items():
            if col in df.columns:
                df.at[row_idx, col] = value
    
    # 5. Salvar planilha
    print(f"\n[*] Salvando planilha...")
    df.to_excel(PLANILHA_PATH, index=False)
    print(f"   [OK] Planilha salva: {PLANILHA_PATH}")
    
    # 6. Resumo
    print(f"\n[*] RESUMO:")
    print(f"   Total de registros: {len(df)}")
    print(f"   Classificações preenchidas: {df['Classificação_LLM'].notna().sum()}")
    print(f"   Erros: {df['Erro'].notna().sum() if 'Erro' in df.columns else 0}")
    
    total_input_tokens = df["Input_Tokens"].sum()
    total_output_tokens = df["Output_Tokens"].sum()
    total_tokens = total_input_tokens + total_output_tokens
    avg_latencia = df["Latência"].mean()
    
    print(f"\n[*] TOKENS E LATENCIA:")
    print(f"   Total Input Tokens: {int(total_input_tokens):,}")
    print(f"   Total Output Tokens: {int(total_output_tokens):,}")
    print(f"   Total Tokens: {int(total_tokens):,}")
    print(f"   Latencia Media: {avg_latencia:.0f}ms")
    
    print("\n" + "=" * 80)
    print("[OK] CONCLUSAO: Batch processing concluido!")
    print("=" * 80)


if __name__ == "__main__":
    main()
