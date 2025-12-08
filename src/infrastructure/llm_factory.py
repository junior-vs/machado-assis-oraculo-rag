from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config import settings

class LLMFactory:
    @staticmethod
    def get_llm():
        return ChatGoogleGenerativeAI(
            model=settings.model_name, 
            temperature=settings.temperature,
            api_key=settings.gemini_api_key
        )

    @staticmethod
    def get_embeddings():
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key # pyright: ignore[reportArgumentType]
        )