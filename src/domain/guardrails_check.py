from pydantic import BaseModel, Field


class InputGuardrail(BaseModel):
    is_valid: bool = Field(
        description="A pergunta é válida para o contexto de Dom Casmurro? True ou False"
    )
    reason: str = Field(
        default="",
        description="Explicação curta se a pergunta for inválida (ex: tema fora do livro)."
    )


class RetrievalGrader(BaseModel):
    binary_score: str = Field(
        description="O documento contém a resposta? 'sim' ou 'nao'"
    )

class HallucinationGrade(BaseModel):
    binary_score: str = Field(
        description="A resposta é apoiada pelos fatos fornecidos? 'sim' ou 'nao'"
    )
    reason: str = Field(
        description="Explicação breve de por que a resposta é ou não é alucinação."
    )