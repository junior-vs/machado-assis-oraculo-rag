#!/usr/bin/env python
"""
Script de teste para validar o pipeline RAG completo
"""
import sys
from pathlib import Path

from src.infrastructure.vector_store import VectorStoreRepository
from src.use_cases.graph import RAGGraphBuilder

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_rag_pipeline():
    """Testa o pipeline RAG com questões de exemplo"""
    
    print("🔬 Testando Pipeline RAG Completo...")
    print("="*60)
    
    # Inicializar repositório de vetores
    print("\n1️⃣ Inicializando VectorStore...")
    try:
        repo = VectorStoreRepository()
        retriever = repo.get_retriever()
        print("   ✅ VectorStore inicializado")
    except Exception as e:
        print(f"   ❌ Erro ao inicializar: {e}")
        return

    # Construir grafo
    print("\n2️⃣ Construindo grafo RAG...")
    try:
        graph_builder = RAGGraphBuilder(retriever)
        app = graph_builder.build()
        print("   ✅ Grafo construído com sucesso")
    except Exception as e:
        print(f"   ❌ Erro ao construir grafo: {e}")
        return

    # Perguntas de teste
    test_questions = [
        "Quem é Bento Santiago em Dom Casmurro?",
        "Qual é o tema central de Dom Casmurro?",
        "Quem é Capitu em Dom Casmurro?",
    ]

    # Testar cada pergunta
    for i, question in enumerate(test_questions, 1):
        print(f"\n3.{i}️⃣ Testando pergunta: '{question}'")
        print("-"*60)
        
        try:
            # Preparar estado inicial
            state = {
                "question": question,
                "documents": [],
                "generation": "",
                "loop_count": 0,
            }
            
            # Invocar grafo
            print("   ⏳ Processando...")
            final_state = app.invoke(state)
            
            # Exibir resultado
            print(f"\n   📖 Resposta:\n   {final_state['generation']}")
            print(f"   📊 Documentos usados: {len(final_state.get('documents', []))}")
            print(f"   🔄 Iterações: {final_state.get('loop_count', 0)}")
            
        except Exception as e:
            print(f"   ❌ Erro ao processar: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ Testes completos!")

if __name__ == "__main__":
    test_rag_pipeline()
