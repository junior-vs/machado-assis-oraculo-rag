# Machado Oráculo

A modular Retrieval-Augmented Generation (RAG) system with automatic query correction, powered by LangChain and LangGraph. Answers literary questions about Machado de Assis' masterpiece "Memórias Póstumas de Brás Cubas" with intelligent document grading and query refinement.

## Features

- **Corrective RAG Pipeline**: Automatically evaluates document relevance and reformulates queries when needed
- **Vector-Based Retrieval**: Fast similarity search using FAISS with semantic embeddings
- **LLM-Powered Grading**: Uses Claude or Gemini to assess document relevance to user questions
- **Query Refinement**: When initial documents are irrelevant, the system rewords the query for better results
- **Clean Architecture**: Well-separated layers (domain, infrastructure, use_cases) for maintainability
- **CLI Interface**: Interactive chat loop for testing and interaction
- **Type-Safe**: Full type hints and Pydantic models for robust validation

## Quick Start

### Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) (modern Python package installer)
- Google Generative AI API key (or OpenAI key)
- Internet connection (for downloading corpus)

### Installation

1. Clone the repository and navigate to the project:

```bash
cd machado_oraculo
```

2. Install dependencies and create a virtual environment with `uv`:

```bash
uv sync
```

This command automatically creates a virtual environment and installs all dependencies from `pyproject.toml`.

3. Configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
GEMINI_API_KEY=your-api-key-here
MODEL_NAME=gemini-2.5-flash
TEMPERATURE=0.0
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

4. Initialize the system:

```bash
uv run python initialize.py
```

This script validates your configuration and creates the vector store index.

## Usage

### Interactive CLI

Run the interactive chat interface:

```bash
uv run python -m src.main
```

Example questions:

```
🗣️  Sua pergunta: Quem é Brás Cubas?
🗣️  Sua pergunta: O que é Memórias Póstumas de Brás Cubas?
🗣️  Sua pergunta: Qual é o estilo literário de Machado de Assis?
```

Type `sair`, `exit`, or `quit` to close the application.

### Programmatic Use

```python
from src.infrastructure.vector_store import VectorStoreRepository
from src.use_cases.graph import RAGGraphBuilder

# Initialize the system
repo = VectorStoreRepository()
retriever = repo.get_retriever()

# Build and run the RAG pipeline
graph_builder = RAGGraphBuilder(retriever)
app = graph_builder.build()

# Execute a query
question = "Quem é Brás Cubas?"
inputs = {"question": question, "loop_count": 0}

for output in app.stream(inputs):
    print(output)
```

## How It Works

The system implements a **Corrective RAG** workflow with four stages:

```
User Question
     ↓
┌─────────────────────────┐
│ RETRIEVE                │ ← Search for relevant documents
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ GRADE DOCUMENTS         │ ← LLM evaluates relevance
└────────────┬────────────┘
             ↓
       ┌─────┴──────┐
       │            │
    RELEVANT    NOT RELEVANT
       │            │
       ↓            ↓
    GENERATE   TRANSFORM_QUERY
       │     (rewrite question)
       │            │
       └─────┬──────┘
             ↓
       [Loop with limit]
             ↓
┌─────────────────────────┐
│ GENERATE                │ ← Create final answer
└────────────┬────────────┘
             ↓
        Final Answer
```

### Stage Details

1. **Retrieve**: Queries the vector store to find up to 3 semantically similar documents
2. **Grade**: Uses an LLM to determine if retrieved documents are relevant to the question
3. **Transform Query**: If documents aren't relevant, reformulates the question for better retrieval
4. **Generate**: Creates the final answer using relevant documents or provides a fallback response

The pipeline loops up to 3 times before generating an answer, ensuring quality results.

## Project Structure

```
machado_oraculo/
├── src/
│   ├── config.py                 # Configuration management with Pydantic
│   ├── main.py                   # CLI entry point
│   ├── domain/
│   │   └── state.py              # TypedDict state schema for the workflow
│   ├── infrastructure/
│   │   ├── llm_factory.py        # LLM and embeddings factory (singleton)
│   │   └── vector_store.py       # FAISS vector store repository
│   └── use_cases/
│       ├── nodes.py              # RAG nodes (retrieve, grade, generate, transform)
│       └── graph.py              # LangGraph workflow builder
├── initialize.py                 # Setup script for vectorstore initialization
├── main.py                       # Entry point for running with python -m src.main
├── test_rag.py                   # Integration tests
├── test_retriever.py             # Retriever tests
├── test_grader.py                # Document grader tests
├── pyproject.toml                # Project metadata and dependencies
└── README.md                     # This file
```

## Configuration

Settings are managed through `src/config.py` and loaded from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Generative AI API key |
| `MODEL_NAME` | `gemini-2.5-flash` | LLM model to use |
| `TEMPERATURE` | `0.0` | LLM temperature for deterministic responses |
| `CHUNK_SIZE` | `1000` | Document chunk size for splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `BOOK_URL` | Project Gutenberg URL | Source corpus URL |
| `STORAGE_PATH` | `machado.txt` | Local storage for downloaded corpus |
| `FAISS_INDEX_PATH` | `vectorstore` | Directory for FAISS index |

## Testing

Run the test suite:

```bash
# All tests
uv run pytest

# Specific test file
uv run pytest test_rag.py -v

# Integration tests only
uv run python test_rag.py
```

## Architecture Highlights

### Clean Architecture
- **Domain**: State definitions and business logic contracts
- **Infrastructure**: External integrations (LLM, vector store)
- **Use Cases**: Orchestration and workflow implementation

### Design Patterns
- **Factory Pattern**: Centralized LLM and embeddings creation
- **Repository Pattern**: Abstraction over vector store operations
- **State Machine**: LangGraph manages workflow orchestration
- **Singleton**: Shared LLM instance to optimize API usage

### Type Safety
- Full type hints throughout the codebase
- Pydantic models for configuration and LLM structured output
- TypedDict for immutable state passing

## Requirements

### Runtime
- Python 3.13+
- 500 MB disk space (corpus + index)
- API access to Google Generative AI (or OpenAI)

### Development
- pytest for testing
- A code editor with Python support (VS Code, PyCharm, etc.)

## Troubleshooting

### API Key Issues

```
❌ ERRO: GEMINI_API_KEY não configurada no .env
```

**Solution**: Ensure your `.env` file contains a valid `GEMINI_API_KEY`.

### Vector Store Initialization Fails

```
❌ ERRO ao inicializar vectorstore
```

**Solution**: Check your internet connection and ensure the corpus URL is accessible. Delete the `vectorstore/` directory and run `python initialize.py` again.

### Out of Memory

**Solution**: Reduce `CHUNK_SIZE` in `.env` or run on a machine with more RAM. The corpus is approximately 500 MB when expanded.

## Performance

- **Retrieval**: <100ms (FAISS similarity search)
- **Grading**: 1-3 seconds per document (LLM call)
- **Generation**: 2-5 seconds (LLM call)
- **Total latency**: 5-15 seconds per query (including retry loops)

## API Keys

### Google Generative AI

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the Generative AI API
4. Create an API key
5. Add to `.env`

### OpenAI (Alternative)

Modify `src/config.py` to use `ChatOpenAI` from `langchain_openai` instead of `ChatGoogleGenerativeAI`.

## Future Enhancements

- [ ] Support for multiple corpus sources
- [ ] Fine-tuned embedding models
- [ ] Streaming responses with real-time feedback
- [ ] Document citation and source attribution
- [ ] Batch processing for multiple questions
- [ ] Web interface with FastAPI

## License

This project is part of the "AI para Devs" course series and follows the repository's license.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Machado de Assis - Wikipedia](https://en.wikipedia.org/wiki/Machado_de_Assis)
- [RAG Overview](https://python.langchain.com/docs/use_cases/question_answering/)
- ✅ Baixa e processa o corpus de Machado de Assis

## 💬 Uso

### Interactive CLI

```bash
uv run python -m src.main
```

Exemplos de perguntas:

```
🗣️  Sua pergunta: Quem é Brás Cubas?
🗣️  Sua pergunta: Qual é o tema central de Quincas Borba?
🗣️  Sua pergunta: Fale sobre o pessimismo em Machado de Assis
```

### Programático

```bash
uv run python
```

Then in the Python REPL:

```python
from src.main import run_rag_pipeline

result = run_rag_pipeline("Quem é Brás Cubas?")
print(result['generation'])
```

## 🔄 Fluxo do RAG

1. **RETRIEVE**: Busca 4 documentos (chunks) similares à pergunta usando FAISS
2. **GRADE**: LLM avalia cada documento com pergunta: "Este documento responde a pergunta?"
3. **GERA OU TRANSFORMA**:
   - Se há documentos relevantes → vai para GENERATE
   - Se não → reformula pergunta com TRANSFORM_QUERY
4. **LOOP**: Retentar até limite (default: 3 tentativas)
5. **GENERATE**: LLM cria resposta usando documentos relevantes

## 📊 Configurações

No arquivo `.env`, você pode ajustar:

```env
# Modelo e comportamento
MODEL_NAME=gpt-3.5-turbo          # ou gpt-4o-mini, gpt-4
TEMPERATURE=0.0                   # 0 = determinístico, 1 = criativo

# Processamento de documentos
CHUNK_SIZE=1000                   # Tamanho dos chunks em caracteres
CHUNK_OVERLAP=200                 # Sobreposição entre chunks

# Fonte de dados
BOOK_URL=...                      # URL do corpus (default: Machado de Assis)
STORAGE_PATH=machado.txt          # Arquivo local para armazenar corpus

# Recuperação
RETRIEVER_K=4                     # Número de documentos a recuperar

# RAG Corretivo
MAX_LOOP_COUNT=3                  # Máximo de reescritas de pergunta
```

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Execute como módulo Python:

```bash
python -m src.main
```

### `openai.AuthenticationError`

- Verifique se `OPENAI_API_KEY` está correto em `.env`
- Asegure-se que a chave ainda é válida

### `FAISS index not found`

Execute `python initialize.py` novamente para baixar e indexar o corpus.

### Respostas vagas ou irrelevantes

- Aumentar `RETRIEVER_K` em `.env` para 6-8
- Reduzir `CHUNK_SIZE` de 1000 para 500
- Usar modelo mais forte: `gpt-4o` em vez de `gpt-3.5-turbo`

## 📈 Performance

- **Primeira execução**: ~30-60s (download do corpus)
- **Consultas normais**: ~5-10s por pergunta
- **Limite de reescritas**: 3 tentativas máximo

## 🔐 Segurança

- Nunca commitar `.env` com chaves reais
- `.env.example` serve como template
- As chaves são carregadas via `python-dotenv`

## 📚 Estrutura do Projeto

```
machado_oraculo/
├── src/
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings
│   ├── main.py                # CLI interativa
│   │
│   ├── domain/
│   │   ├── __init__.py
│   │   └── state.py           # GraphState TypedDict
│   │
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── llm_factory.py     # ChatOpenAI factory
│   │   └── vector_store.py    # FAISS + embeddings
│   │
│   └── use_cases/
│       ├── __init__.py
│       ├── nodes.py           # Nós do grafo (retrieve, grade, generate, transform)
│       └── graph.py           # LangGraph builder
│
├── docs/
│   └── Aula3_rag.ipynb        # Notebook original (referência)
│
├── .env.example               # Template de configuração
├── .gitignore                 # Exclui .env, vectorstore, etc.
├── pyproject.toml             # Dependências do projeto
├── initialize.py              # Script de setup
└── README.md                  # Este arquivo
```

## 🔍 Como Funciona o RAG Corretivo

O sistema implementa o padrão "Corrective RAG":

1. **Retrieval**: Busca inicial baseada em similaridade
2. **Grading**: LLM avalia se documentos são úteis
3. **Correction**: Se não houver documentos bons, reformula a pergunta
4. **Generation**: Cria resposta com contexto dos melhores documentos

Diferente de RAG simples, este sistema **não aceita documentos ruins** - ele tenta reescrever a pergunta até encontrar relevância ou atingir o limite de tentativas.

## 📝 Logs e Debug

Para ver mais detalhes durante execução, adicione ao `main.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contribuições

Este projeto foi migrado de um notebook Jupyter (`docs/Aula3_rag.ipynb`) para uma arquitetura modular e escalável.

## 📄 Licença

Ver arquivo LICENSE (se aplicável)

---

**Desenvolvido com ❤️ para estudar RAG e LangChain**
