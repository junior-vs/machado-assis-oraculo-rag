from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    gemini_api_key: str
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    book_url: str = "https://www.gutenberg.org/files/55752/55752-0.txt"
    storage_path: str = "machado.txt"
    faiss_index_path: str = "vectorstore"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings() # pyright: ignore[reportCallIssue]