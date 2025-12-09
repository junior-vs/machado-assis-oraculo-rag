from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from src.domain.state import GraphState
from src.domain.guardrails_check import InputGuardrail, RetrievalGrader
from src.infrastructure.llm_factory import LLMFactory
from src.utils.logging import get_logger


logger = get_logger()

class RAGNodes:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = LLMFactory.get_llm()
        self.grader_chain = self._build_grader_chain()
        self.rag_chain = self._build_rag_chain()
        self.rewriter_chain = self._build_rewriter_chain()
        self.guardrail_chain = self._build_guardrail_chain()



    def _build_grader_chain(self):
        # Usa function_calling com gpt-3.5-turbo (não suporta json_schema)
        llm_structured = self.llm.with_structured_output(RetrievalGrader, method="function_calling")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em avaliar relevância de documentos. 
Sua tarefa é determinar se um documento recuperado responde ou é relevante para a pergunta do usuário.

Critérios de relevância:
- O documento contém informações sobre o tópico?
- O documento pode ajudar a responder a pergunta?
- Existe qualquer conexão temática?

Responda 'sim' se houver qualquer relevância, 'nao' apenas se for completamente irrelevante."""),
            ("human", "Pergunta: {question}\n\nDocumento recuperado:\n{document}\n\nEste documento é relevante para a pergunta? Responda 'sim' ou 'nao'.")
        ])
        return prompt | llm_structured

    def _build_rag_chain(self):
        prompt = PromptTemplate(
            template="""Você é um especialista em Machado de Assis. Use o contexto: {context} 
            para responder à pergunta: {question}.""",
            input_variables=["context", "question"]
        )
        return prompt | self.llm | StrOutputParser()

    def _build_rewriter_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em reformular perguntas sobre "Dom Casmurro" de Machado de Assis.
Sua tarefa é reescrever a pergunta do usuário mantendo seu significado ORIGINAL, mas usando terminologia e contexto do livro.

Dicas:
- Mantenha o sentido original da pergunta
- Use nomes de personagens, temas e conceitos do livro quando apropriado
- Reescreva de forma mais clara e específica para melhorar a busca
- Não mude a intenção da pergunta, apenas refine-a

Pergunta original: {original_question}

Reescreva de forma mais clara e específica para busca sobre o livro:"""),
            ("human", "{question}")
        ])
        return prompt | self.llm | StrOutputParser()

    # --- NÓS DO GRAFO ---
    
    def retrieve(self, state: GraphState):
        logger.debug(f"Buscando documentos para: {state['question'][:500]}...")
        documents = self.retriever.invoke(state["question"])
        logger.info(f"Recuperados {len(documents)} documentos")
        return {"documents": documents, "question": state["question"]}

    def grade_documents(self, state: GraphState):
        logger.debug("Avaliando relevância dos documentos...")
        question = state["question"]
        documents = state["documents"]
        
        relevant_docs = []
        for i, doc in enumerate(documents):
            logger.debug(f"Avaliando documento {i+1}/{len(documents)}")
            try:
                score = self.grader_chain.invoke({
                    "question": question, 
                    "document": doc.page_content
                })
                is_relevant = score.binary_score.lower() == "sim"
                logger.debug(f"Documento {i+1}: {'RELEVANTE' if is_relevant else 'NÃO RELEVANTE'}")
                
                if is_relevant:
                    relevant_docs.append(doc)
            except Exception as e:
                logger.warning(f"Erro ao avaliar documento {i+1}: {e}")
        
        logger.info(f"Documentos relevantes: {len(relevant_docs)}/{len(documents)}")
        return {"documents": relevant_docs, "question": question}

    def generate(self, state: GraphState):
        logger.debug("Gerando resposta...")
        context_text = "\n\n".join([d.page_content for d in state["documents"]])
        
        generation = self.rag_chain.invoke({
            "context": context_text, 
            "question": state["question"]
        })
        
        logger.info("Resposta gerada com sucesso")
        return {"generation": generation}

    def transform_query(self, state: GraphState):
        logger.debug(f"Reescrevendo pergunta (tentativa {state.get('loop_count', 0) + 1})")
        # Usar pergunta original como base (armazenada no estado)
        original_question = state.get("original_question", state["question"])
        new_q = self.rewriter_chain.invoke({
            "original_question": original_question,
            "question": state["question"]
        })
        logger.info(f"Pergunta reescrita: \n {new_q[:500]}...")
        return {"question": new_q, "loop_count": state.get("loop_count", 0) + 1}

 

    # 2. Construção da Chain de Guardrail
    def _build_guardrail_chain(self):
        #llm_structured = self.llm.with_structured_output(GradeDocuments, method="function_calling")
        llm_structured = self.llm.with_structured_output(InputGuardrail, method="function_calling")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um guardião de conhecimento sobre o livro 'Dom Casmurro' de Machado de Assis.
            Sua função é filtrar perguntas que contenham premissas falsas, erros factuais graves ou que sejam sobre outros livros/assuntos.
            
            Exemplos de REJEIÇÃO:
            - "Quem é Bento Santiago em Dom Casmurro?" (Correto)
            - "Quem é Capitu?" (Correto, personagem de Dom Casmurro)
            - "Qual é o tema de Dom Casmurro?" (Correto)
            - "Qual a receita de bolo de cenoura?" (Fora do contexto)
            
            Exemplos de APROVAÇÃO:
            - "Quem foi Virgília?"
            - "Por que ele dedicou o livro ao verme?"
            
            Analise a pergunta e determine se ela é válida para processamento.
            Analise se a pergunta é válida (sobre o livro, sem erros factuais graves).
    
            Retorne:
                - is_valid: true se a pergunta é válida, false caso contrário
                - binary_score: 'sim' se válida, 'não' se inválida
                - reason: explicação breve se rejeitada
            
            """),
            ("human", "Pergunta: {question}")
        ])
        return prompt | llm_structured

    # ... (Manter métodos existentes: retrieve, grade_documents, etc) ...

    # 3. Novo Nó do Grafo
    def guardrails_check(self, state: GraphState):
        logger.debug("🛡️ Verificando Guardrails da pergunta...")
        question = state["question"]
        
        try:
            outcome = self.guardrail_chain.invoke({"question": question})
            
            if outcome.is_valid:
                logger.info("✅ Pergunta aprovada pelo Guardrail.")
                return {"question": question, "generation": None} # Continua normal
            else:
                logger.warning(f"⛔ Pergunta bloqueada: {outcome.reason}")
                # Definimos a 'generation' com a recusa para exibir ao utilizador
                return {
                    "question": question, 
                    "generation": f"Não posso responder a isso. {outcome.reason}"
                }
                
        except Exception as e:
            logger.error(f"Erro no guardrail: {e}")
            # Em caso de erro técnico, optamos por "fail open" (deixar passar) ou "fail closed".
            # Aqui vamos deixar passar para não travar o sistema.
            return {"question": question}