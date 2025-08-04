from dotenv import load_dotenv
import os, sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)

class ModelLoader:
    def __init__(self) -> None:
        load_dotenv()
        self._validate_env()
        self.config = load_config("config/config.yaml")
        log.info("Configuration loaded successfully.", config_keys = list(self.config.keys()))

    def _validate_env(self):
        """
        Validate that the required environment variables are set.
        Ensure API keys exist.
        """
        required_vars = ["OPENAI_API_KEY", "GROQ_API_KEY"]
        self.api_keys = {var: os.getenv(var) for var in required_vars}
        missing_vars = [var for var, value in self.api_keys.items() if value is None]
        if missing_vars:
            log.error("Missing environment variables: %s", missing_vars)
            raise DocumentPortalException("Missing environment variables", sys) #type: ignore
        log.info("Environment variables validated successfully.", api_keys=list(self.api_keys.keys()))

    def load_embeddings(self):
        """
        Load and return the embedding model.
        """
        try:
            log.info("Loading OpenAI embeddings model.")
            model_name = self.config["embeddings_model"]["model_name"]
            return OpenAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Error loading model embeddings", error=str(e))
            raise DocumentPortalException("Error loading model embeddings", sys) #type: ignore

    def load_llm(self):
        """
        Load and return the language model.
        """

        llm_block = self.config["llm"]

        log.info("Loading LLM")
        provider_key = os.getenv("LLM_PROVIDER", "openai")

        if provider_key not in llm_block:
            log.error("LLM provider not found in configuration", provider_key=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in configuration")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature")
        max_tokens = llm_config.get("max_tokens")

        if provider == "groq":
            llm = ChatGroq(
                model=model_name,
                api_key=self.api_keys["GROQ_API_KEY"], #type: ignore
                temperature=temperature,
                max_tokens=max_tokens
            )
            return llm
        
        elif provider == "openai":
            llm = ChatOpenAI(
                model=model_name,
                api_key=self.api_keys["OPENAI_API_KEY"], #type: ignore
                temperature=temperature
            )
            return llm
        
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")

    