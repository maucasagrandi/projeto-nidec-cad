"""
Componentes para exibição elegante de JSON estruturado no Streamlit.
Ideal para apresentar dados técnicos a clientes.
"""

import streamlit as st
import json
from typing import Any, Dict


def display_json_card(
    data: Dict[str, Any],
    title: str,
    icon: str = "📋",
    color: str = "#1f77e1",
) -> None:
    """
    Exibe JSON em um card elegante com syntax highlighting.
    
    Args:
        data: Dicionário ou Pydantic model (será convertido)
        title: Título do card
        icon: Emoji para o título
        color: Cor da borda (hex)
    """
    # Converte Pydantic model para dict se necessário
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # CSS customizado para o card
    card_css = f"""
    <div style="
        border: 2px solid {color};
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #0e1117;
    ">
        <div style="color: {color}; font-size: 18px; font-weight: bold; margin-bottom: 10px;">
            {icon} {title}
        </div>
        <pre style="
            background-color: #161b22;
            color: #c9d1d9;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.5;
        ">{json_str}</pre>
    </div>
    """
    
    st.markdown(card_css, unsafe_allow_html=True)
    
    # Botão para copiar
    st.text_area("Copie o JSON abaixo:", value=json_str, height=100, disabled=True, key=f"json_{title}")


def display_json_comparison(
    data_before: Dict[str, Any],
    data_after: Dict[str, Any],
    title_before: str = "Antes",
    title_after: str = "Depois",
) -> None:
    """
    Exibe dois JSONs lado-a-lado para comparação.
    
    Args:
        data_before: Dados do lado esquerdo
        data_after: Dados do lado direito
        title_before: Título do lado esquerdo
        title_after: Título do lado direito
    """
    # Converte Pydantic models se necessário
    if hasattr(data_before, 'model_dump'):
        data_before = data_before.model_dump()
    if hasattr(data_after, 'model_dump'):
        data_after = data_after.model_dump()
    
    json_before = json.dumps(data_before, indent=2, ensure_ascii=False)
    json_after = json.dumps(data_after, indent=2, ensure_ascii=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {title_before}")
        st.code(json_before, language="json")
    
    with col2:
        st.markdown(f"### {title_after}")
        st.code(json_after, language="json")


def display_json_expandable(
    data: Dict[str, Any],
    title: str,
    icon: str = "📂",
) -> None:
    """
    Exibe JSON em expander para economizar espaço.
    
    Args:
        data: Dicionário ou Pydantic model
        title: Título do expander
        icon: Emoji para o título
    """
    # Converte Pydantic model se necessário
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    
    with st.expander(f"{icon} {title}", expanded=False):
        st.json(data)


def display_json_table(
    data_list: list,
    title: str,
    icon: str = "📊",
) -> None:
    """
    Exibe lista de JSONs como tabela formatada.
    
    Args:
        data_list: Lista de dicionários
        title: Título da tabela
        icon: Emoji para o título
    """
    st.write(f"{icon} **{title}**")
    
    if not data_list:
        st.info("Nenhum item para exibir")
        return
    
    # Converte para dicts se necessário
    data_list = [
        item.model_dump() if hasattr(item, 'model_dump') else item
        for item in data_list
    ]
    
    # Cria DataFrame
    import pandas as pd
    df = pd.DataFrame(data_list)
    
    # Exibe com styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def display_json_summary(
    data: Dict[str, Any],
    title: str = "Resumo",
    highlight_fields: list = None,
) -> None:
    """
    Exibe JSON em formato de resumo com highlights.
    
    Args:
        data: Dicionário
        title: Título do resumo
        highlight_fields: Lista de campos para destacar
    """
    # Converte Pydantic model se necessário
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    
    highlight_fields = highlight_fields or []
    
    st.write(f"### {title}")
    
    for key, value in data.items():
        if key in highlight_fields:
            # Destaca em cor
            st.markdown(f"**✨ {key}:** `{value}`")
        else:
            st.markdown(f"**{key}:** `{value}`")


def display_json_schema(
    pydantic_model,
    title: str = "Schema da Estrutura",
) -> None:
    """
    Exibe JSON Schema de um Pydantic model.
    
    Args:
        pydantic_model: Classe Pydantic
        title: Título
    """
    schema = pydantic_model.model_json_schema()
    
    with st.expander(f"📋 {title}", expanded=False):
        st.json(schema)


def display_json_metrics(
    data: Dict[str, Any],
    title: str = "Métricas",
    columns: int = 3,
) -> None:
    """
    Exibe campos de um JSON como métricas do Streamlit.
    
    Args:
        data: Dicionário com valores métricos
        title: Título
        columns: Número de colunas
    """
    # Converte Pydantic model se necessário
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    
    st.write(f"### {title}")
    
    cols = st.columns(columns)
    
    for idx, (key, value) in enumerate(data.items()):
        with cols[idx % columns]:
            st.metric(label=key.replace('_', ' ').title(), value=value)


def create_json_report(
    title: str,
    sections: list,
) -> str:
    """
    Cria um relatório HTML completo com JSONs formatados.
    Retorna HTML string para download.
    
    Args:
        title: Título do relatório
        sections: Lista de dicts com {title, data, icon}
    
    Returns:
        HTML string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f6f8fa;
            }}
            h1 {{
                color: #0366d6;
                border-bottom: 3px solid #0366d6;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #24292e;
                margin-top: 30px;
                margin-bottom: 15px;
                border-left: 4px solid #0366d6;
                padding-left: 10px;
            }}
            .section {{
                background-color: white;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            pre {{
                background-color: #f6f8fa;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
                font-size: 12px;
                line-height: 1.45;
            }}
            code {{
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                color: #24292e;
            }}
            .metric {{
                display: inline-block;
                background-color: #f6f8fa;
                border-radius: 6px;
                padding: 10px 15px;
                margin: 5px;
                border: 1px solid #e1e4e8;
            }}
            .icon {{
                font-size: 1.5em;
                margin-right: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """
    
    for section in sections:
        icon = section.get('icon', '📋')
        section_title = section.get('title', 'Seção')
        data = section.get('data', {})
        
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        html += f"""
        <div class="section">
            <h2><span class="icon">{icon}</span>{section_title}</h2>
            <pre><code>{json_str}</code></pre>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    return html
