"""
Creator (Factory) pattern for creating LLM and Embedding models.
This module provides a structured way to instantiate different model providers.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ollama_client import AuthenticatedOllamaEmbeddings, CustomOllamaChat


class ModelCreator(ABC):
    """Abstract base class for model creators."""

    @abstractmethod
    def create_llm(self) -> Optional[BaseChatModel]:
        """Creates a chat model instance."""
        pass

    @abstractmethod
    def create_embeddings(self) -> Optional[Embeddings]:
        """Creates an embeddings model instance."""
        pass


class OllamaModelCreator(ModelCreator):
    """Creator for Ollama models."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not self.base_url:
            print("   - Ollama 비활성화됨 (OLLAMA_BASE_URL이 빈 값으로 설정됨). 건너뜁니다.")

    def create_llm(self) -> Optional[BaseChatModel]:
        if not self.base_url:
            return None

        print("   - Ollama Chat 모델 초기화를 시도합니다.")
        api_key = os.getenv("LLM_API_KEY")  # Ollama Gateway용 API 키
        chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
        
        llm = CustomOllamaChat(
            base_url=self.base_url, chat_model=chat_model, temperature=0.0, api_key=api_key
        )
        print("   - Ollama Chat 모델 초기화 완료.")
        return llm

    def create_embeddings(self) -> Optional[Embeddings]:
        if not self.base_url:
            return None
            
        print("   - Ollama Embeddings 모델 초기화를 시도합니다.")
        api_key = os.getenv("LLM_API_KEY")
        embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")

        embeddings = AuthenticatedOllamaEmbeddings(
            base_url=self.base_url, model=embedding_model, api_key=api_key
        )
        print("   - Ollama Embeddings 모델 초기화 완료.")
        return embeddings


class FabrixModelCreator(ModelCreator):
    """Creator for FabriX (OpenAI-compatible) models."""

    def __init__(self):
        self.base_url = os.getenv("FABRIX_BASE_URL")
        self.api_key = os.getenv("FABRIX_API_KEY")
        if not (self.base_url and self.api_key):
            print("   - FabriX 설정 없음 (FABRIX_BASE_URL 또는 FABRIX_API_KEY). 건너뜁니다.")

    def create_llm(self) -> Optional[BaseChatModel]:
        if not (self.base_url and self.api_key):
            return None

        print("   - FabriX Chat 모델 초기화를 시도합니다 (OpenAI 호환 모드).")
        chat_model = os.getenv("FABRIX_CHAT_MODEL", "fabrix-chat-model")
        
        llm = ChatOpenAI(
            model=chat_model, temperature=0.0, api_key=self.api_key, base_url=self.base_url
        )
        print("   - FabriX Chat 모델 초기화 완료.")
        return llm

    def create_embeddings(self) -> Optional[Embeddings]:
        if not (self.base_url and self.api_key):
            return None

        print("   - FabriX Embeddings 모델 초기화를 시도합니다 (OpenAI 호환 모드).")
        embedding_model = os.getenv("FABRIX_EMBEDDING_MODEL", "fabrix-embed-model")

        embeddings = OpenAIEmbeddings(
            model=embedding_model, api_key=self.api_key, base_url=self.base_url
        )
        print("   - FabriX Embeddings 모델 초기화 완료.")
        return embeddings


def get_model_creator(provider_name: str) -> Optional[ModelCreator]:
    """Factory function to get a model creator based on the provider name."""
    creators = {"ollama": OllamaModelCreator, "fabrix": FabrixModelCreator}
    creator_class = creators.get(provider_name.lower())
    return creator_class() if creator_class else None