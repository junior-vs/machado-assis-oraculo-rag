#!/usr/bin/env python
"""
Script de diagnóstico para testar grader diretamente
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.use_cases.nodes import RAGNodes
from src.infrastructure.vector_store import VectorStoreRepository

# Setup
repo = VectorStoreRepository()
retriever = repo.get_retriever(k=3)
nodes = RAGNodes(retriever)

# Teste de grading
question = "Quem é Bento Santiago em Dom Casmurro?"
docs = retriever.invoke(question)

print("=" * 70)
print("TESTE DE GRADER")
print("=" * 70)
print(f"\nPergunta: '{question}'")
print(f"Documentos recuperados: {len(docs)}\n")

for i, doc in enumerate(docs, 1):
    content = doc.page_content[:150]
    print(f"\n[{i}] Documento: {content}...")
    print("-" * 70)
    
    try:
        # Invoca o grader diretamente
        score = nodes.grader_chain.invoke({
            "question": question, 
            "document": doc.page_content
        })
        
        print(f"    Tipo de retorno: {type(score)}")
        print(f"    Valor bruto: {score}")
        
        # Tenta extrair o valor
        if isinstance(score, dict):
            score_value = score.get("binary_score", "")
        else:
            score_value = getattr(score, "binary_score", "")
        
        print(f"    Score extraído: '{score_value}'")
        print(f"    Relevante: {'SIM' if score_value.lower() == 'sim' else 'NAO'}")
        
    except Exception as e:
        print(f"    ERRO: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
