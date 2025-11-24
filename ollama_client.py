"""
Ollama 엔드포인트 및 호출 관련 클래스
폐쇄망 환경에서 Ollama와의 통신을 관리합니다.
"""

import os
import json
import inspect
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
        
        # 기본 URL 구성 (/api/embed 또는 /api/embeddings)
        # OllamaEmbeddings는 /api/embed를 사용할 수 있으므로 둘 다 시도
        urls_to_try = [
            f"{self.base_url}/api/embed",
            f"{self.base_url}/api/embeddings"
        ]
        
        # 요청 헤더 구성
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 각 텍스트에 대해 임베딩 요청
        embeddings = []
        for idx, text in enumerate(texts):
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            last_error = None
            embedding_found = False
            
            for url in urls_to_try:
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Ollama 응답 형식에 따라 임베딩 추출
                    embedding = None
                    if isinstance(result, dict):
                        if "embedding" in result:
                            embedding = result["embedding"]
                        elif "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                            embedding = result["data"][0].get("embedding", [])
                        else:
                            # 첫 번째 리스트 값 찾기
                            for value in result.values():
                                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
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
    
    def get_chat_llm(self) -> "LLMAPIClient":
        """
        채팅용 LLM 인스턴스 반환 (자기 자신 반환)
        
        Returns:
            LLMAPIClient: 자기 자신 인스턴스
        """
        return self
    
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


class LLMAPIClient(LLM):
    """
    HTTP request 방식으로 LLM 서버 endpoint를 호출하는 클래스
    LLM 클래스를 상속받아 구현
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
        api_endpoint: str,
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
        # 환경 변수에서 API 키 가져오기
        final_api_key = api_key or os.getenv("LLM_API_KEY")
        final_base_url = os.getenv("OLLAMA_BASE_URL", base_url)
        
        # LLM 초기화 (Pydantic 모델이므로 super() 사용)
        super().__init__(
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
        # Ollama /api/chat 형식으로 변환
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ollama_messages.append({"role": "assistant", "content": msg.content})
            else:
                ollama_messages.append({"role": "user", "content": str(msg.content)})
        
        # /api/chat 엔드포인트 사용
        chat_endpoint = self.api_endpoint.replace("/generate", "/chat")
        if "/chat" not in chat_endpoint:
            # 엔드포인트가 /api/generate인 경우 /api/chat으로 변경
            chat_endpoint = chat_endpoint.replace("/api/generate", "/api/chat")
            if "/api/chat" not in chat_endpoint:
                chat_endpoint = f"{self.base_url}/api/chat"
        
        # 요청 페이로드 구성
        payload = {
            "model": self.chat_model,
            "messages": ollama_messages,
            "temperature": self.temperature,
            "stream": False,
            **kwargs
        }
        
        # Tools가 바인딩되어 있으면 tools 정보를 payload에 추가
        if hasattr(self, "_bound_tools") and self._bound_tools:
            # LangChain tools를 Ollama API 형식에 맞게 변환
            formatted_tools = []
            for tool_obj in self._bound_tools:
                # Tool의 스키마 추출
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_obj.name,
                        "description": tool_obj.description or "",
                    }
                }
                
                # Tool의 파라미터 스키마 추가
                parameters = None
                
                # 방법 1: LangChain tool의 input_schema 사용 (Pydantic v2)
                if parameters is None and hasattr(tool_obj, "input_schema"):
                    try:
                        input_schema = tool_obj.input_schema
                        if hasattr(input_schema, "model_json_schema"):
                            parameters = input_schema.model_json_schema()
                        elif hasattr(input_schema, "schema"):
                            # Pydantic v1 호환
                            try:
                                parameters = input_schema.schema()
                            except Exception:
                                # CallableSchema 등의 경우 스킵
                                pass
                    except Exception as e:
                        pass  # 조용히 실패
                
                # 방법 2: args_schema가 있는 경우 (Pydantic BaseModel)
                if parameters is None and hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    try:
                        args_schema = tool_obj.args_schema
                        # Pydantic v2
                        if hasattr(args_schema, "model_json_schema"):
                            parameters = args_schema.model_json_schema()
                        # Pydantic v1
                        elif hasattr(args_schema, "schema"):
                            try:
                                schema_result = args_schema.schema()
                                # CallableSchema 체크 (문자열로 확인)
                                if isinstance(schema_result, dict) and "type" in schema_result:
                                    # 유효한 JSON 스키마인 경우만 사용
                                    if schema_result.get("type") != "callable":
                                        parameters = schema_result
                            except (TypeError, ValueError) as e:
                                # CallableSchema 등의 경우 스킵
                                if "CallableSchema" not in str(e):
                                    pass
                    except Exception as e:
                        pass  # 조용히 실패
                
                # 방법 3: tool의 함수에서 직접 파라미터 추출
                if parameters is None:
                    try:
                        # tool 객체의 실제 함수 가져오기
                        func = None
                        if hasattr(tool_obj, "func"):
                            func = tool_obj.func
                        elif hasattr(tool_obj, "_func"):
                            func = tool_obj._func
                        elif callable(tool_obj):
                            func = tool_obj
                        
                        if func:
                            sig = inspect.signature(func)
                            properties = {}
                            required = []
                            
                            for param_name, param in sig.parameters.items():
                                if param_name == "self":
                                    continue
                                
                                param_type = "string"  # 기본값
                                if param.annotation != inspect.Parameter.empty:
                                    ann_str = str(param.annotation)
                                    if "int" in ann_str:
                                        param_type = "integer"
                                    elif "float" in ann_str:
                                        param_type = "number"
                                    elif "bool" in ann_str:
                                        param_type = "boolean"
                                    elif "list" in ann_str.lower() or "List" in ann_str:
                                        param_type = "array"
                                
                                properties[param_name] = {
                                    "type": param_type,
                                    "description": f"Parameter {param_name}"
                                }
                                
                                # 기본값이 없으면 required에 추가
                                if param.default == inspect.Parameter.empty:
                                    required.append(param_name)
                            
                            if properties:
                                parameters = {
                                    "type": "object",
                                    "properties": properties
                                }
                                if required:
                                    parameters["required"] = required
                    except Exception as e:
                        pass  # 조용히 실패
                
                # 방법 4: 기본 파라미터 스키마 생성
                if parameters is None:
                    parameters = {
                        "type": "object",
                        "properties": {}
                    }
                
                # JSON 직렬화 가능한지 확인하고 정리
                try:
                    # JSON 직렬화 테스트
                    json.dumps(parameters)
                    tool_schema["function"]["parameters"] = parameters
                except (TypeError, ValueError) as e:
                    print(f"경고: parameters JSON 직렬화 실패, 기본 스키마 사용: {e}")
                    tool_schema["function"]["parameters"] = {
                        "type": "object",
                        "properties": {}
                    }
                
                formatted_tools.append(tool_schema)
            
            payload["tools"] = formatted_tools
            payload["tool_choice"] = "auto"  # 자동으로 tool 선택
        
        # 헤더 복사 (API Key 포함)
        headers = self._headers.copy() if hasattr(self, "_headers") else {}
        
        # API Key가 있으면 Authorization 헤더에 추가
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # Payload JSON 직렬화 가능한지 확인
            try:
                json.dumps(payload)
            except (TypeError, ValueError) as e:
                print(f"경고: Payload JSON 직렬화 실패, kwargs 제거 후 재시도: {e}")
                # kwargs에 문제가 있을 수 있으므로 제거
                payload_clean = {
                    "model": payload["model"],
                    "messages": payload["messages"],
                    "temperature": payload["temperature"],
                    "stream": payload["stream"]
                }
                if "tools" in payload:
                    payload_clean["tools"] = payload["tools"]
                if "tool_choice" in payload:
                    payload_clean["tool_choice"] = payload["tool_choice"]
                payload = payload_clean
            
            # HTTP POST 요청
            response = requests.post(
                chat_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 응답 파싱
            result = response.json()
            
            # Tool calls 추출
            tool_calls = []
            content = None
            
            # Ollama /api/chat 응답 형식에서 메시지 추출
            if isinstance(result, dict):
                if "message" in result:
                    message = result["message"]
                    if isinstance(message, dict):
                        # Content 추출
                        if "content" in message:
                            content = message["content"]
                        elif "text" in message:
                            content = message["text"]
                        
                        # Tool calls 추출
                        if "tool_calls" in message:
                            tool_calls = message["tool_calls"]
                        elif "tool_calls" in result:
                            tool_calls = result["tool_calls"]
                    else:
                        content = str(message)
                elif "response" in result:
                    content = result["response"]
                elif "content" in result:
                    content = result["content"]
                else:
                    # 첫 번째 문자열 값 반환
                    for key, value in result.items():
                        if isinstance(value, str):
                            content = value
                            break
                    if content is None:
                        content = str(result)
            elif isinstance(result, str):
                content = content or result
            else:
                content = str(result)
            
            # Tool calls 추출 및 변환
            langchain_tool_calls = []
            if tool_calls:
                # Ollama 형식의 tool_calls를 LangChain 형식으로 변환
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function = tool_call.get("function", {})
                        if function:
                            # arguments가 문자열인 경우 JSON 파싱
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
            
            # AIMessage 생성 (tool_calls 포함)
            if langchain_tool_calls:
                ai_message = AIMessage(
                    content=content or "",
                    tool_calls=langchain_tool_calls
                )
            else:
                ai_message = AIMessage(content=content or "")
            
            return ai_message
                
        except requests.exceptions.RequestException as e:
            raise ValueError(f"LLM API 호출 실패: {str(e)}")
    
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "LLMAPIClient":
        """
        Tools를 바인딩한 새로운 LLMAPIClient 인스턴스 반환
        
        Args:
            tools: 바인딩할 도구 리스트
            **kwargs: 추가 파라미터
            
        Returns:
            LLMAPIClient: Tools가 바인딩된 새로운 인스턴스
        """
        # 새 인스턴스 생성 (tools 정보 저장)
        new_instance = LLMAPIClient(
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
            "model": self.chat_model,
            "temperature": self.temperature,
            "stream": False,
            **kwargs
        }
        
        if stop:
            payload["stop"] = stop
        
        # 헤더 복사 (API Key 포함)
        headers = self._headers.copy() if hasattr(self, "_headers") else {}
        
        # API Key가 있으면 Authorization 헤더에 추가
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            # HTTP POST 요청
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
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
    
    def get_embeddings(self) -> OllamaEmbeddings:
        """
        임베딩 모델 인스턴스 반환
        
        Returns:
            OllamaEmbeddings: 임베딩용 Ollama 모델 인스턴스
        """
        # API Key가 있으면 커스텀 헤더를 포함한 OllamaEmbeddings 래퍼 사용
        if self.api_key:
            return AuthenticatedOllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
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

