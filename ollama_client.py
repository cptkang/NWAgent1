"""
Ollama 엔드포인트 및 호출 관련 클래스
폐쇄망 환경에서 Ollama와의 통신을 관리합니다.
"""

import os
from typing import Optional

# langchain-ollama 패키지 사용 (bind_tools 지원)
from langchain_ollama import ChatOllama, OllamaEmbeddings



class OllamaClient:
    """Ollama 모델과의 통신을 관리하는 클래스"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        chat_model: str = "llama3.1",
        embedding_model: str = "mxbai-embed-large",
        temperature: float = 0.0
    ):
        """
        Ollama 클라이언트 초기화
        
        Args:
            base_url: Ollama 서버 주소 (기본값: http://localhost:11434)
            chat_model: 채팅용 모델 이름
            embedding_model: 임베딩용 모델 이름
            temperature: LLM 생성 온도 값
        """
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        
        # 환경 변수에서 Ollama URL 가져오기 (있는 경우)
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url)
    
    def get_chat_llm(self) -> ChatOllama:
        """
        채팅용 LLM 인스턴스 반환
        
        Returns:
            ChatOllama: 채팅용 Ollama 모델 인스턴스
        """
        return ChatOllama(
            model=self.chat_model,
            base_url=self.base_url,
            temperature=self.temperature
        )
    
    def get_embeddings(self) -> OllamaEmbeddings:
        """
        임베딩 모델 인스턴스 반환
        
        Returns:
            OllamaEmbeddings: 임베딩용 Ollama 모델 인스턴스
        """
        return OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.base_url
        )
    
    def update_chat_model(self, model_name: str) -> None:
        """
        채팅 모델 변경
        
        Args:
            model_name: 새로운 모델 이름
        """
        self.chat_model = model_name
    
    def update_embedding_model(self, model_name: str) -> None:
        """
        임베딩 모델 변경
        
        Args:
            model_name: 새로운 모델 이름
        """
        self.embedding_model = model_name

