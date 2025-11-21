"""
Ollama 엔드포인트 및 호출 관련 클래스
폐쇄망 환경에서 Ollama와의 통신을 관리합니다.
"""

import os
import json
import requests
from typing import Optional, Any, List, Dict

# langchain-ollama 패키지 사용 (bind_tools 지원)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import LLMResult, ChatGeneration, ChatResult
from langchain_core.language_models.base import LanguageModelInput



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


class LLMAPIClient(OllamaClient, LLM):
    """
    HTTP request 방식으로 LLM 서버 endpoint를 호출하는 클래스
    OllamaClient를 상속받고 LLM 클래스를 상속받아 구현
    """
    
    def __init__(
        self,
        api_endpoint: str,
        base_url: str = "http://localhost:11434",
        chat_model: str = "llama3.1",
        embedding_model: str = "mxbai-embed-large",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60
    ):
        """
        LLMAPIClient 초기화
        
        Args:
            api_endpoint: LLM 서버 API endpoint URL
            base_url: 기본 URL (OllamaClient 호환성)
            chat_model: 채팅용 모델 이름
            embedding_model: 임베딩용 모델 이름
            temperature: LLM 생성 온도 값
            api_key: API 키 (선택사항)
            headers: 추가 HTTP 헤더 (선택사항)
            timeout: 요청 타임아웃 (초)
        """
        # OllamaClient 초기화
        OllamaClient.__init__(
            self,
            base_url=base_url,
            chat_model=chat_model,
            embedding_model=embedding_model,
            temperature=temperature
        )
        
        # LLM 초기화
        LLM.__init__(self, temperature=temperature)
        
        self.api_endpoint = api_endpoint
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.timeout = timeout
        
        # 기본 헤더 설정
        self.headers = {
            "Content-Type": "application/json",
            **(headers or {})
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    @property
    def _llm_type(self) -> str:
        """LLM 타입 식별자"""
        return "llm_api_client"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        LLM 클래스의 필수 메서드: 프롬프트를 받아서 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            stop: 중지 시퀀스 리스트
            run_manager: 콜백 매니저
            **kwargs: 추가 파라미터
            
        Returns:
            str: LLM 응답 텍스트
        """
        return self._call_http(prompt, stop=stop, **kwargs)
    
    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> BaseMessage:
        """
        ChatModel 호환성을 위한 메서드: 메시지 리스트를 받아서 응답 생성
        
        Args:
            messages: 메시지 리스트
            **kwargs: 추가 파라미터
            
        Returns:
            BaseMessage: AI 응답 메시지
        """
        # 메시지를 프롬프트로 변환
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")
            else:
                prompt_parts.append(str(msg.content))
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # LLM 호출
        response_text = self._call(prompt, **kwargs)
        
        # AIMessage 반환
        return AIMessage(content=response_text)
    
    def _call_http(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        HTTP 요청으로 LLM 서버 호출
        
        Args:
            prompt: 입력 프롬프트
            stop: 중지 시퀀스 리스트
            **kwargs: 추가 파라미터
            
        Returns:
            str: LLM 응답 텍스트
        """
        # 요청 페이로드 구성
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            **kwargs
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            # HTTP POST 요청
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 응답 파싱
            result = response.json()
            
            # 응답 형식에 따라 텍스트 추출
            if isinstance(result, dict):
                if "text" in result:
                    return result["text"]
                elif "content" in result:
                    return result["content"]
                elif "response" in result:
                    return result["response"]
                elif "output" in result:
                    return result["output"]
                elif "message" in result:
                    if isinstance(result["message"], dict) and "content" in result["message"]:
                        return result["message"]["content"]
                    return str(result["message"])
                else:
                    # 첫 번째 문자열 값 반환
                    for key, value in result.items():
                        if isinstance(value, str):
                            return value
                    return str(result)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"LLM API 호출 실패: {str(e)}")
    
    def get_chat_llm(self) -> "LLMAPIClient":
        """
        채팅용 LLM 인스턴스 반환 (자기 자신 반환)
        
        Returns:
            LLMAPIClient: 자기 자신 인스턴스
        """
        return self

