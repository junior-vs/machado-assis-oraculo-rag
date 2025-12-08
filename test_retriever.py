#!/usr/bin/env python
"""
Script de diagnóstico para testar retriever diretamente
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure.vector_store import VectorStoreRepository

# Teste de retrieval
repo = VectorStoreRepository()
retriever = repo.get_retriever(k=5)

questions = [
    "Quem é Brás Cubas?",
    "Machado de Assis",
    "romance brasileiro",
]

print("=" * 70)
print("TESTE DE RETRIEVER")
print("=" * 70)

for q in questions:
    print(f"\n📌 Pergunta: '{q}'")
    print("-" * 70)
    
    docs = retriever.invoke(q)
    print(f"   Documentos retornados: {len(docs)}")
    
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:200]
        print(f"\n   [{i}] {content}...")
