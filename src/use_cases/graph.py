from langgraph.graph import StateGraph, END
from src.domain.state import GraphState
from src.use_cases.nodes import RAGNodes

class RAGGraphBuilder:
    def __init__(self, retriever):
        self.nodes = RAGNodes(retriever)

# 1. Nova lógica condicional
    def _check_guardrail_result(self, state: GraphState):
        # Se já tivermos uma "generation" nesta etapa, significa que o guardrail rejeitou
        # e preencheu a resposta com o motivo do erro.
        if state.get("generation"):
            return "end" # Encerra o fluxo
        return "retrieve" # Continua para a busca

    def _store_original_question(self, state: GraphState):
        """Armazena a pergunta original no estado para referência durante reescrita."""
        if "original_question" not in state:
            return {"original_question": state["question"]}
        return state

    def _decide_next_step(self, state: GraphState):
        if not state["documents"]:
            if state.get("loop_count", 0) > 3:
                print("🛑 Limite de tentativas excedido.")
                return "generate" # Fallback
            return "transform_query"
        return "generate"

    def build(self):
        workflow = StateGraph(GraphState)

        # Adiciona nós
        workflow.add_node("store_question", self._store_original_question)
        workflow.add_node("guardrails", self.nodes.guardrails_check) # <--- NOVO NÓ
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("transform_query", self.nodes.transform_query)

        # Define fluxo
        workflow.set_entry_point("store_question")
        
        # Fluxo alterado: store -> guardrails -> (decisão)
        workflow.add_edge("store_question", "guardrails")
        
        # Condicional após Guardrails
        workflow.add_conditional_edges(
            "guardrails",
            self._check_guardrail_result,
            {
                "end": END,        # Se falhar no guardrail, sai direto
                "retrieve": "retrieve" # Se passar, busca documentos
            }
        )

        # O resto segue igual
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_next_step,
            {
                "transform_query": "transform_query",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()