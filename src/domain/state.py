from typing import List, Any
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Estado que representa o fluxo de conhecimento no grafo RAG.
    """

    question: str
    generation: str
    documents: List[Any]  # Lista de documentos (Document objects)
    loop_count: int  # Contador para evitar loops infinitos
    max_loops: int  # Número máximo de iterações permitidas
    