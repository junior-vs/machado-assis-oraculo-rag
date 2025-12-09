from langgraph.graph import StateGraph, END
from src.domain.state import GraphState
from src.use_cases.nodes import RAGNodes
from langgraph.checkpoint.memory import MemorySaver  # <--- Importante para a Memória

class RAGGraphBuilder:
    def __init__(self, retriever):
        self.nodes = RAGNodes(retriever)

# NOVA Lógica Condicional para o Output Guardrail
    def _check_hallucination(self, state: GraphState):
        # Se detectou alucinação
        if state.get("hallucination", False):
            loop_count = state.get("loop_count", 0)
            
            # Se ainda temos tentativas, vamos tentar de novo (transform_query)
            if loop_count <= 3: # Limite hardcoded ou use state['max_loops'] se tiver
                print("🔄 Alucinação detectada. Tentando reformular e buscar novamente...")
                return "transform_query"
            else:
                # Se acabou as tentativas, entregamos com um aviso (ou poderíamos sobrescrever a resposta)
                print("🛑 Limite de tentativas excedido com alucinação.")
                return "end"
        
        return "end"

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

        # Adiciona nós (Mantém os anteriores e adiciona o novo)
        workflow.add_node("store_question", self._store_original_question)
        workflow.add_node("guardrails", self.nodes.guardrails_check)
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("validate_gen", self.nodes.validate_generation) # <--- NOVO NÓ
        workflow.add_node("transform_query", self.nodes.transform_query)

        # Fluxo
        workflow.set_entry_point("store_question")
        
        workflow.add_edge("store_question", "guardrails")
        
        workflow.add_conditional_edges(
            "guardrails",
            self._check_guardrail_result,
            {
                "end": END,
                "retrieve": "retrieve"
            }
        )

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
        
        # MUDANÇA AQUI: Generate não vai mais para END, vai para Validate
        workflow.add_edge("generate", "validate_gen")
        
        # Condicional após Validação
        workflow.add_conditional_edges(
            "validate_gen",
            self._check_hallucination,
            {
                "transform_query": "transform_query", # Tenta corrigir
                "end": END                            # Aceita ou desiste
            }
        )
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)