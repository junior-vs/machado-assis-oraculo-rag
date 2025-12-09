"""
Módulo de logging centralizado com suporte a estruturação e auditoria.
"""

import sys
from pathlib import Path
from loguru import logger


class LoggingManager:
    """Gerenciador centralizado de logging com suporte a múltiplos níveis."""
    
    _initialized = False
    _log_level = "INFO"
    
    @classmethod
    def setup(cls, log_level: str = "INFO", audit: bool = False) -> None:
        """Configura o sistema de logging."""
        if cls._initialized:
            return
            
        cls._log_level = log_level
        logger.remove()
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan> - "  # Remove :function e :line
                "<level>{message}</level>"
            ),
            colorize=True,
        )
        
        logger.add(
            log_dir / "app.log",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} \n {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )
        
        if audit:
            logger.add(
                log_dir / "audit.jsonl",
                level="INFO",
                format="{message}",
                serialize=True,
                rotation="10 MB",
                retention="90 days",
            )
        
        cls._initialized = True
        logger.info(f"Sistema de logging inicializado (nível: {log_level})")
    
    @classmethod
    def get_logger(cls):
        """Retorna a instância global do logger."""
        return logger


def get_logger():
    """Obtém o logger global."""
    return LoggingManager.get_logger()

