
1. Estrutura de pastas

```plain-text
.
│
├── .env                    # Suas chaves de API (não comitar)
├── requirements.txt        # Dependências
├── main.py                 # Ponto de entrada (CLI)
│
└── src/
    ├── __init__.py
    ├── config.py           # Configurações globais (Pydantic)
    │
    ├── domain/             # Entidades e Definições de Estado
    │   ├── __init__.py
    │   └── state.py        # TypedDict do LangGraph
    │
    ├── infrastructure/     # Ferramentas externas (FAISS, OpenAI, Web)
    │   ├── __init__.py
    │   ├── llm_factory.py  # Criação dos modelos
    │   └── vector_store.py # Ingestão e setup do FAISS
    │
    └── use_cases/          # A lógica de negócio (O Grafo)
        ├── __init__.py
        ├── nodes.py        # As funções de cada nó (Retrieve, Generate, Grade)
        └── graph.py        # A montagem do workflow LangGraph
```
