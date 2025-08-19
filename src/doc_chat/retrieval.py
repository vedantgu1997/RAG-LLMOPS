import sys
import os
from operator import itemgetter
from typing import Optional, List
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt_library.prompts import PROMPT_REGISTRY
from model.models import PromptType

class ConversationalRAG:
    def __init__(self, session_id: str, retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.llm = self._load_llm()
            self.contextualize_prompt = PROMPT_REGISTRY.get(PromptType.CONTEXTUALIZE_QUESTION.value)
            self.qa_prompt = PROMPT_REGISTRY.get(PromptType.CONTEXT_QA.value)
            if retriever is None:
                raise ValueError("Retriever must be provided for ConversationalRAG")
            self.retriever = retriever
            self._build_lcel_chain()
            self.log.info("ConversationalRAG initialized successfully.", session_id=self.session_id)


        except Exception as e:
            self.log.error('Failed to initialize ConversationalRAG', error=str(e))
            raise DocumentPortalException("Failed to initialize ConversationalRAG", sys)  # type: ignore

    def load_retriever_from_faiss(self, index_path: str):
        '''
        Load the retriever from a FAISS index.
        '''
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")
            
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            self.retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={"k": 5})
            self.log.info("Retriever loaded from FAISS index successfully.", index_path=index_path, session_id=self.session_id)
            return self.retriever
            
        except Exception as e:
            self.log.error("Error loading retriever from FAISS", error=str(e))
            raise DocumentPortalException("Error loading retriever from FAISS", sys)  # type: ignore

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]]) -> str:
        try:
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning("No answer returned from ConversationalRAG", user_input=user_input, session_id=self.session_id)
                return "No relevant information found."
            self.log.info("ConversationalRAG invoked successfully.", user_input=user_input, session_id=self.session_id, answer_preview=answer[:100])
            return answer
        except Exception as e:
            self.log.error("Error invoking ConversationalRAG", error=str(e))
            raise DocumentPortalException("Error invoking ConversationalRAG", sys)  # type: ignore

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("Failed to load LLM")
            self.log.info("LLM loaded successfully.", model_name=llm.model_name, session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error("Error loading LLM", error=str(e))
            raise DocumentPortalException("Error loading LLM in ConversationalRAG", sys) # type: ignore

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        try:
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.contextualize_prompt # type: ignore
                | self.llm
                | StrOutputParser()
            )

            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt # type: ignore
                | self.llm
                | StrOutputParser()
            )
            self.log.info("LCEL chain built successfully.", session_id=self.session_id)
        except Exception as e:
            self.log.error("Error building LCEL chain", error=str(e))
            raise DocumentPortalException("Error building LCEL chain in ConversationalRAG", sys)  # type: ignore