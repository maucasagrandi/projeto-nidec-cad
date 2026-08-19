"""
Módulo para logging de custos e performance de análises CAD.
Salva tokens utilizados e latência em custos.csv
"""

import csv
import os
from datetime import datetime
from typing import Optional
from src.modeling.llm_models import AnalysisMetadata


class CostLogger:
    """Logger para custos e métricas de performance"""
    
    def __init__(self, csv_path: str = "custos.csv"):
        self.csv_path = csv_path
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Inicializa o arquivo CSV com headers se não existir"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'modelo',
                    'total_tokens',
                    'prompt_tokens',
                    'completion_tokens',
                    'latencia_ms',
                    'custo_usd'
                ])
    
    def calculate_cost(self, metadata: AnalysisMetadata) -> float:
        """
        Calcula custo estimado em USD para o Gemini 2.5 Flash Image
        
        Preços (conforme GCP Vertex AI - pode variar):
        - Entrada: $0.075 por 1M tokens
        - Saída: $0.30 por 1M tokens
        """
        # Preços por milhão de tokens
        input_price_per_1m = 0.075
        output_price_per_1m = 0.30
        
        # Calcula custo
        input_cost = (metadata.prompt_tokens / 1_000_000) * input_price_per_1m
        output_cost = (metadata.completion_tokens / 1_000_000) * output_price_per_1m
        
        total_cost = input_cost + output_cost
        return total_cost
    
    def log_analysis(self, metadata: AnalysisMetadata, page_number: int = 1) -> None:
        """
        Registra uma análise no CSV
        
        Args:
            metadata: AnalysisMetadata com tokens e latência
            page_number: Número da página analisada (para referência)
        """
        cost_usd = self.calculate_cost(metadata)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                metadata.timestamp,
                metadata.model,
                metadata.total_tokens,
                metadata.prompt_tokens,
                metadata.completion_tokens,
                f"{metadata.latency_ms:.2f}",
                f"{cost_usd:.6f}"
            ])
    
    def get_summary(self) -> dict:
        """
        Retorna sumário de todos os custos registrados
        """
        if not os.path.exists(self.csv_path):
            return {
                'total_analyses': 0,
                'total_tokens': 0,
                'total_cost': '$0.0000',
                'avg_latency_ms': '0.00',
                'file_path': self.csv_path
            }
        
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        count = 0
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row and row.get('total_tokens'):  # Garante que row existe
                        try:
                            total_tokens += int(row['total_tokens'])
                            total_cost += float(row['custo_usd'])
                            total_latency += float(row['latencia_ms'])
                            count += 1
                        except (ValueError, KeyError) as e:
                            # Ignora linhas malformadas
                            continue
        except Exception as e:
            print(f"Erro ao ler custos.csv: {e}")
            return {
                'total_analyses': 0,
                'total_tokens': 0,
                'total_cost': '$0.0000',
                'avg_latency_ms': '0.00',
                'file_path': self.csv_path
            }
        
        return {
            'total_analyses': count,
            'total_tokens': total_tokens,
            'total_cost': f"${total_cost:.4f}",
            'avg_latency_ms': f"{total_latency / count:.2f}" if count > 0 else '0.00',
            'file_path': self.csv_path
        }
