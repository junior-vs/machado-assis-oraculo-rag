import sys
import argparse
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.vector_store import VectorStoreRepository
from src.use_cases.graph import RAGGraphBuilder
from src.utils.logging import LoggingManager, get_logger  # ← ADICIONE ESTA LINHA


def parse_arguments():
    """Processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Machado Oráculo - Sistema RAG Corretivo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  uv run python -m src.main                    # INFO (padrão)
  uv run python -m src.main --debug            # DEBUG (muito detalhado)
  uv run python -m src.main --warning          # WARNING (apenas avisos+)
  uv run python -m src.main --debug --audit    # DEBUG com auditoria JSON
        """
    )
    
    # Grupo mutuamente exclusivo para nível de log
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Ativar logging DEBUG (muito detalhado)"
    )
    log_group.add_argument(
        "-i", "--info",
        action="store_true",
        help="Ativar logging INFO (padrão)"
    )
    log_group.add_argument(
        "-w", "--warning",
        action="store_true",
        help="Ativar logging WARNING (apenas avisos e erros)"
    )
    log_group.add_argument(
        "-e", "--error",
        action="store_true",
        help="Ativar logging ERROR (apenas erros críticos)"
    )
    
    # Flag de auditoria
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Ativar logging estruturado para auditoria (JSON)"
    )
    
    return parser.parse_args()


def determine_log_level(args) -> str:
    """Determina o nível de logging baseado nos argumentos."""
    if args.debug:
        return "DEBUG"
    elif args.info:
        return "INFO"
    elif args.warning:
        return "WARNING"
    elif args.error:
        return "ERROR"
    else:
        return "INFO"  # Padrão


def main():
    """Função principal com suporte a logging estruturado."""
    # Processar argumentos
    args = parse_arguments()
    log_level = determine_log_level(args)
    
    # Inicializar logging
    LoggingManager.setup(log_level=log_level, audit=args.audit)
    logger = get_logger()
    
    # Log inicial
    logger.info(f"{'='*60}")
    logger.info(f"Inicializando Assistente Literário (Corrective RAG)")
    logger.info(f"Nível de logging: {log_level}")
    logger.info(f"Auditoria estruturada: {'ATIVA' if args.audit else 'INATIVA'}")
    logger.info(f"{'='*60}")
    
    # 1. Setup da Infraestrutura
    try:
        logger.debug("Inicializando Vector Store...")
        repo = VectorStoreRepository()
        retriever = repo.get_retriever()
        logger.info("✅ Vector Store inicializado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao inicializar banco de dados: {e}", exc_info=True)
        return

    # 2. Construção do Grafo
    try:
        logger.debug("Construindo grafo RAG...")
        graph_builder = RAGGraphBuilder(retriever)
        app = graph_builder.build()
        logger.info("✅ Grafo RAG construído com sucesso")
    except Exception as e:
        logger.error(f"Erro ao construir grafo: {e}", exc_info=True)
        return

    logger.info("\n✅ Sistema pronto! (Digite 'sair' para encerrar)")
    logger.info("="*50)

    # 3. Loop de Interação (CLI)
    query_count = 0
    while True:
        try:
            user_input = input("\n🗣️  Sua pergunta: ").strip()
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                logger.info("👋 Até logo!")
                break
            
            if not user_input:
                continue

            query_count += 1
            logger.info(f"[QUERY #{query_count}] \nPergunta: {user_input}")
            print("-" * 30)
            
            # Executar grafo
            inputs = {"question": user_input, "loop_count": 0}
            
            final_state = app.invoke(inputs)
            
            # Log da resposta com estrutura
            logger.info(
                f"[QUERY #{query_count}] Resposta gerada",
                extra={
                    "docs_count": len(final_state.get('documents', [])),
                    "iterations": final_state.get('loop_count', 0)
                }
            )
            
            print(f"\n🤖 Resposta: {final_state['generation']}")
            print("="*50)

        except KeyboardInterrupt:
            logger.warning("Interrupção do usuário (Ctrl+C)")
            print("\n👋 Encerrando...")
            break
        except Exception as e:
            logger.error(f"Erro durante execução: {e}", exc_info=True)


if __name__ == "__main__":
    main()