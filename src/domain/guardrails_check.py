from pydantic import BaseModel, Field

# Modelo para o Guardrail (Filtro de Entrada)
class InputGuardrail(BaseModel):
    is_valid: bool = Field(
        description="A pergunta é válida para o contexto de Dom Casmurro? True ou False"
    )
    reason: str = Field(
        default="",
        description="Explicação curta se a pergunta for inválida (ex: tema fora do livro)."
    )

# Modelo para o Grader (Avaliação de Documentos recuperados)
class RetrievalGrader(BaseModel):
    binary_score: str = Field(
        description="O documento contém a resposta? 'sim' ou 'nao'"
    )