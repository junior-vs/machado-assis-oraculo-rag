from typing import List, Any, Optional, Tuple
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Estado que representa o fluxo de conhecimento no grafo RAG.
    """

    # --- Campos Principais ---
    question: str           # A pergunta atual (pode ser reescrita)
    generation: str         # A resposta gerada pelo LLM
    documents: List[Any]    # Lista de documentos recuperados (LangChain Documents)
    
    # --- Controle de Fluxo ---
    loop_count: int         # Contador para evitar loops infinitos
    max_loops: int          # Número máximo de iterações permitidas
    original_question: Optional[str] # A pergunta original do usuário (para referência)

    # --- Novos Campos (Guardrails & Memória) ---
    hallucination: bool     # Flag: True se a resposta foi considerada alucinação
    chat_history: List[Tuple[str, str]] # Histórico de mensagens (role, message) para memória (Buffer)