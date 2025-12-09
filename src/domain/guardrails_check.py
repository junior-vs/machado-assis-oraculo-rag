from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documento é relevante para a pergunta? 'sim' ou 'não'"
    )
    is_valid: bool = Field(
        description="A pergunta é válida para Dom Casmurro? True ou False"
    )
    reason: str = Field(
        default="",
        description="Razão pela qual a pergunta foi rejeitada (se aplicável)"
    )