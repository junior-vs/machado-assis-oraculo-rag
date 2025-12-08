#!/usr/bin/env python3
"""
Script para inicializar o vectorstore com o corpus de Machado de Assis.
Execute isto uma vez antes de usar o sistema RAG.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.infrastructure.vector_store import VectorStoreRepository
from src.infrastructure.llm_factory import LLMFactory


def initialize():
    """Inicializa o vectorstore e testa a conexão com a OpenAI"""
    print("🔧 Inicializando Sistema Machado Oráculo...\n")
    
    # 1. Valida configurações
    print("✅ Verificando configurações...")
    if not settings.gemini_api_key:
        print("❌ ERRO: GEMINI_API_KEY não configurada no .env")
        return False
    print("✅ GEMINI_API_KEY encontrada")
    
    # 2. Testa conexão com LLM
    print("\n🤖 Testando conexão com Google Gemini...")
    try:
        _ = LLMFactory.get_llm()
        print("✅ Conexão com Google Gemini estabelecida")
    except Exception as e:
        print(f"❌ ERRO ao conectar com Google Gemini: {e}")
        return False
    
    # 3. Inicializa vectorstore
    print("\n📚 Inicializando Vectorstore...")
    try:
        vs_repo = VectorStoreRepository()
        print("✅ Vectorstore inicializado com sucesso")
        print(f"   📂 Armazenado em: {settings.faiss_index_path}")
    except Exception as e:
        print(f"❌ ERRO ao inicializar vectorstore: {e}")
        return False
    
    # 4. Testa retriever
    print("\n🔍 Testando retriever...")
    try:
        _ = vs_repo.get_retriever()
        print("✅ Retriever funcional")
    except Exception as e:
        print(f"❌ ERRO ao testar retriever: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ SISTEMA PRONTO!")
    print("="*60)
    print("\nAgora você pode executar:")
    print("  python -m src.main")
    print("\n")
    return True


if __name__ == "__main__":
    success = initialize()
    sys.exit(0 if success else 1)
