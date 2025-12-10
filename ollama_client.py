"""
Ollama 엔드포인트 및 호출 관련 클래스
폐쇄망 환경에서 Ollama와의 통신을 관리합니다.
"""

import os
import json
import requests
from typing import Optional, Any, List, Dict, Tuple

# langchain-ollama 패키지 사용 (bind_tools 지원)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun, AsyncCallbackManagerForChainRun
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, AIMessageChunk
from langchain_core.outputs import LLMResult, ChatGeneration, ChatResult
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.embeddings import Embeddings
from pydantic import Field


class AuthenticatedOllamaEmbeddings(OllamaEmbeddings):
    """
    API Key 인증을 지원하는 OllamaEmbeddings 래퍼 클래스
    """
    
    api_key: Optional[str] = Field(default=None, description="API Key for authentication")
    
    def __init__(
        self,
        model: str = "mxbai-embed-large",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        AuthenticatedOllamaEmbeddings 초기화
        
        Args:
            model: 임베딩 모델 이름
            base_url: Ollama 서버 주소
            api_key: API Key (선택사항)
            **kwargs: 추가 파라미터
        """
        super().__init__(model=model, base_url=base_url, **kwargs)
        self.api_key = api_key or os.getenv("LLM_API_KEY")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 리스트를 임베딩으로 변환 (API Key 헤더 추가)
        """
        return self._embed_with_auth(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리를 임베딩으로 변환 (API Key 헤더 추가)
        """
        embeddings = self._embed_with_auth([text])
        return embeddings[0] if embeddings else []
    
    def _embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        텍스트를 임베딩으로 변환 (API Key 헤더 추가)
        """
        return self._embed_with_auth(texts)
    
    def _embed_with_auth(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 임베딩으로 변환 (API Key 헤더 추가)
        """
        import requests
        
        # 최신 Ollama는 /api/embed (key: "input"), 구버전은 /api/embeddings (key: "prompt")
        endpoints_to_try = [
            (f"{self.base_url}/api/embed", "input"),
            (f"{self.base_url}/api/embeddings", "prompt"),
        ]
        
        # 요청 헤더 구성
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 각 텍스트에 대해 임베딩 요청
        embeddings = []
        for idx, text in enumerate(texts):
            last_error = None
            embedding_found = False
            
            for url, payload_key in endpoints_to_try:
                payload = {
                    "model": self.model,
                    payload_key: text,
                }

                try:
                    response = requests.post(
                        url, json=payload, headers=headers, timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Ollama 응답 형식에 따라 임베딩 추출
                    embedding = None
                    if isinstance(result, dict):
                        if "embedding" in result:
                            embedding = result["embedding"]
                        elif (
                            "data" in result
                            and isinstance(result["data"], list)
                            and len(result["data"]) > 0
                        ):
                            embedding = result["data"][0].get("embedding", [])
                        else:
                            # 첫 번째 리스트 값 찾기
                            for value in result.values():
                                if (
                                    isinstance(value, list)
                                    and len(value) > 0
                                    and isinstance(value[0], (int, float))
                                ):
                                    embedding = value
                                    break
                    elif isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            embedding = result[0]
                        else:
                            embedding = result
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        embedding_found = True
                        break

                except requests.exceptions.RequestException as e:
                    last_error = e
                    continue
            
            if not embedding_found:
                # 모든 URL 시도 실패
                raise ValueError(f"Embedding API 호출 실패 (텍스트 {idx+1}/{len(texts)}): {last_error}")
        
        return embeddings


class CustomOllamaChat(BaseChatModel):
    """
    HTTP request 방식으로 LLM 서버 endpoint를 호출하는 클래스
    BaseChatModel 클래스를 상속받아 LangGraph와 호환되도록 구현
    """
    
    # Pydantic 필드 선언
    api_endpoint: str
    base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1"
    embedding_model: str = "mxbai-embed-large"
    api_key: Optional[str] = None
    timeout: int = 60
    temperature: float = 0.0
    
    class Config:
        """Pydantic 설정"""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        api_endpoint: str = "/api/chat",
        base_url: str = "http://localhost:11434",
        chat_model: str = "llama3.1",
        embedding_model: str = "mxbai-embed-large",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        **kwargs
    ):
        """
        CustomOllamaChat 초기화
        
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
        # 환경 변수에서 API 키 가져오기
        final_api_key = api_key or os.getenv("LLM_API_KEY")
        final_base_url = os.getenv("OLLAMA_BASE_URL", base_url)
        
        # LLM 초기화 (Pydantic 모델이므로 super() 사용)
        super(BaseChatModel, self).__init__(
            api_endpoint=api_endpoint,
            base_url=final_base_url,
            chat_model=chat_model,
            embedding_model=embedding_model,
            api_key=final_api_key,
            timeout=timeout,
            temperature=temperature,
            **kwargs
        )
        
        # 헤더 설정 (Pydantic 필드가 아닌 일반 속성)
        self._headers = {
            "Content-Type": "application/json",
            **(headers or {})
        }
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
    
    @property
    def _llm_type(self) -> str:
        """LLM 타입 식별자"""
        return "llm_api_client"
    
    def _convert_messages_to_ollama_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """
        LangChain 메시지를 Ollama API 형식으로 변환
        
        Args:
            messages: LangChain 메시지 리스트
            
        Returns:
            Ollama 형식의 메시지 리스트
        """
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ai_dict = {"role": "assistant", "content": msg.content}
                if getattr(msg, "tool_calls", None):
                    ai_dict["tool_calls"] = [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": tc.get("args", {})
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                ollama_messages.append(ai_dict)
            elif isinstance(msg, ToolMessage):
                tool_message = {"role": "tool", "content": str(msg.content)}
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id is not None:
                    tool_message["tool_call_id"] = tool_call_id
                ollama_messages.append(tool_message)
            else:
                ollama_messages.append({"role": "user", "content": str(msg.content)})
        return ollama_messages
    
    def _get_chat_endpoint(self) -> str:
        """
        /api/chat 엔드포인트 URL 반환
        
        Returns:
            chat 엔드포인트 URL
        """
        if self.api_endpoint.startswith("http"):
            return self.api_endpoint
        
        chat_endpoint = f"{self.base_url}{self.api_endpoint}"
        return chat_endpoint
    
    def _format_tools_for_payload(self) -> Optional[List[Dict[str, Any]]]:
        """
        바인딩된 tools를 Ollama API 형식으로 변환
        
        Returns:
            포맷팅된 tools 리스트 또는 None
        """
        if not (hasattr(self, "_bound_tools") and self._bound_tools):
            return None
        
        formatted_tools = []
        for tool_obj in self._bound_tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_obj.name,
                    "description": tool_obj.description or "",
                }
            }
            
            parameters = self._extract_tool_parameters(tool_obj)
            tool_schema["function"]["parameters"] = parameters
            formatted_tools.append(tool_schema)
        
        return formatted_tools
    
    def _extract_tool_parameters(self, tool_obj: Any) -> Dict[str, Any]:
        """
        Tool 객체에서 파라미터 스키마 추출
        
        Args:
            tool_obj: Tool 객체
            
        Returns:
            JSON 스키마 딕셔너리
        """
        if hasattr(tool_obj, "input_schema"):
            try:
                input_schema = tool_obj.input_schema
                if hasattr(input_schema, "model_json_schema"):
                    parameters = input_schema.model_json_schema()
                elif hasattr(input_schema, "schema"):
                    parameters = input_schema.schema()
                else:
                    parameters = None
                
                if parameters:
                    try:
                        json.dumps(parameters)
                        return parameters
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass
        
        return {"type": "object", "properties": {}}
    
    def _build_payload(self, ollama_messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        API 요청 페이로드 구성
        
        Args:
            ollama_messages: Ollama 형식 메시지 리스트
            **kwargs: 추가 파라미터
            
        Returns:
            요청 페이로드 딕셔너리
        """
        payload = {
            "model": self.chat_model,
            "messages": ollama_messages,
            "temperature": self.temperature,
            "stream": False,
            **kwargs
        }
        
        formatted_tools = self._format_tools_for_payload()
        if formatted_tools:
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"
        
        # JSON 직렬화 검증
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as e:
            print(f"경고: Payload JSON 직렬화 실패, kwargs 제거 후 재시도: {e}")
            payload = {
                "model": payload["model"],
                "messages": payload["messages"],
                "temperature": payload["temperature"],
                "stream": payload["stream"]
            }
            if "tools" in payload:
                payload["tools"] = payload["tools"]
            if "tool_choice" in payload:
                payload["tool_choice"] = payload["tool_choice"]
        
        return payload
    
    def _get_request_headers(self) -> Dict[str, str]:
        """
        API 요청 헤더 구성
        
        Returns:
            요청 헤더 딕셔너리
        """
        headers = self._headers.copy() if hasattr(self, "_headers") else {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _parse_response(self, result: Any) -> BaseMessage:
        """
        API 응답을 파싱하여 AIMessage 생성
        
        Args:
            result: API 응답 결과
            
        Returns:
            AIMessage 객체
        """
        content, tool_calls = self._extract_content_and_tool_calls(result)
        langchain_tool_calls = self._convert_tool_calls_to_langchain_format(tool_calls)
        
        if langchain_tool_calls:
            return AIMessage(content=content or "", tool_calls=langchain_tool_calls)
        return AIMessage(content=content or "")
    
    def _extract_content_and_tool_calls(self, result: Any) -> Tuple[Optional[str], List[Any]]:
        """
        응답에서 content와 tool_calls 추출
        
        Args:
            result: API 응답 결과
            
        Returns:
            (content, tool_calls) 튜플
        """
        content = None
        tool_calls = []
        
        if isinstance(result, dict):
            if "message" in result:
                message = result["message"]
                if isinstance(message, dict):
                    content = message.get("content") or message.get("text")
                    tool_calls = message.get("tool_calls", [])
                else:
                    content = str(message)
            else:
                content = result.get("response") or result.get("content")
                if content is None:
                    for value in result.values():
                        if isinstance(value, str):
                            content = value
                            break
                    if content is None:
                        content = str(result)
                tool_calls = result.get("tool_calls", [])
        elif isinstance(result, str):
            content = result
        
        return content, tool_calls
    
    def _convert_tool_calls_to_langchain_format(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Ollama 형식의 tool_calls를 LangChain 형식으로 변환
        
        Args:
            tool_calls: Ollama 형식의 tool_calls 리스트
            
        Returns:
            LangChain 형식의 tool_calls 리스트
        """
        langchain_tool_calls = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            
            function = tool_call.get("function", {})
            if not function:
                continue
            
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    args_dict = json.loads(arguments)
                except json.JSONDecodeError:
                    args_dict = {}
            else:
                args_dict = arguments
            
            langchain_tool_calls.append({
                "name": function.get("name", ""),
                "args": args_dict,
                "id": tool_call.get("id", f"call_{len(langchain_tool_calls)}")
            })
        
        return langchain_tool_calls
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        BaseChatModel의 핵심 메서드: 메시지 리스트를 받아 LLM을 호출하고 결과를 반환
        
        Args:
            messages: 메시지 리스트
            stop: 중단 시퀀스
            run_manager: 콜백 매니저
            **kwargs: 추가 파라미터
        Returns:
            ChatResult: LLM 응답 결과
        """
        ollama_messages = self._convert_messages_to_ollama_format(messages)
        chat_endpoint = self._get_chat_endpoint()
        payload = self._build_payload(ollama_messages, **kwargs)
        headers = self._get_request_headers()
        
        print(f"payload: {payload}")
        
        try:
            response = requests.post(
                chat_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                error_detail = response.text if hasattr(response, "text") else ""
                raise requests.exceptions.HTTPError(f"{e} | body: {error_detail}") from e
            
            result = response.json()
            response_message = self._parse_response(result)
            generation = ChatGeneration(message=response_message)
            return ChatResult(generations=[generation])
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"LLM API 호출 실패: {str(e)}")
    
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "CustomOllamaChat":
        """
        Tools를 바인딩한 새로운 CustomOllamaChat 인스턴스 반환
        
        Args:
            tools: 바인딩할 도구 리스트 (LangChain Tool 또는 Pydantic BaseModel)
            **kwargs: 추가 파라미터
            
        Returns:
            LLMAPIClient: Tools가 바인딩된 새로운 인스턴스
        """
        # 새 인스턴스 생성 (tools 정보 저장)
        new_instance = CustomOllamaChat(
            api_endpoint=self.api_endpoint,
            base_url=self.base_url,
            chat_model=self.chat_model,
            embedding_model=self.embedding_model,
            temperature=self.temperature,
            api_key=self.api_key,
            headers=self._headers.copy() if hasattr(self, "_headers") else None,
            timeout=self.timeout
        )
        
        # tools 정보 저장 (내부적으로 사용)
        new_instance._bound_tools = tools
        new_instance._tool_kwargs = kwargs
        
        return new_instance
