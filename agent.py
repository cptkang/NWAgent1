"""
Agent 클래스
Ollama LLM과 Tools를 결합하여 에이전트를 생성하고 실행합니다.
폐쇄망 환경을 고려하여 LangChain Hub 대신 프롬프트를 직접 정의합니다.
"""

from typing import List, Optional, Dict, Any, Union

# LangChain imports
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import LLM

from ollama_client import OllamaClient, LLMAPIClient
from tools import Tools


class Agent:
    """에이전트 생성 및 실행을 관리하는 클래스"""
    
    def __init__(
        self,
        ollama_client: Union[OllamaClient, LLMAPIClient],
        tools: Tools,
        system_message: str = "너는 IT 및 IT 인프라  전문가이며, 요청에 응답할 때 도구를 사용할 수 있다. 사용자의 질문에 최대한 정확하게 답하고, 필요한 경우 도구를 사용해 계산해라.",
        verbose: bool = True
    ):
        """
        Agent 초기화
        
        Args:
            ollama_client: OllamaClient 또는 LLMAPIClient 인스턴스
            tools: Tools 인스턴스
            system_message: 시스템 메시지
            verbose: 에이전트 실행 과정 출력 여부
        """
        self.ollama_client = ollama_client
        self.tools = tools
        self.system_message = system_message
        self.verbose = verbose
        
        # LLM 및 에이전트 생성
        self.llm = None
        self.agent_executor = None
        
        self._build_agent()
    
    def _build_agent(self) -> None:
        """에이전트 및 에이전트 실행기 생성"""
        print("에이전트 생성 중...")
        
        # LLM 인스턴스 가져오기
        llm_instance = self.ollama_client.get_chat_llm()
        
        # LLMAPIClient인 경우 ChatModel로 래핑
        if isinstance(llm_instance, LLMAPIClient):
            # LLMAPIClient는 invoke 메서드를 구현했으므로 ChatModel처럼 사용 가능
            # create_agent는 BaseChatModel을 기대하지만, LLM도 invoke 메서드가 있으면 작동할 수 있음
            self.llm = llm_instance
        else:
            self.llm = llm_instance
        
        # 도구 리스트 가져오기
        tools_list = self.tools.get_tools()
        print(f"사용 가능한 도구: {[t.name for t in tools_list]}")
        
        # LangChain 1.0+ 버전: create_agent 사용 (tool bind 방식)
        agent_kwargs = {
            "model": self.llm,
            "system_prompt": self.system_message,
            "debug": self.verbose
        }
        
        if tools_list:
            agent_kwargs["tools"] = tools_list
            agent_kwargs["middleware"] = [handle_tool_errors]
        
        self.agent_executor = create_agent(**agent_kwargs)
        print("에이전트 생성 완료")
    
    def invoke(
        self,
        input_text: str,
        chat_history: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        에이전트 실행
        
        Args:
            input_text: 사용자 입력 텍스트
            chat_history: 대화 기록 (선택사항)
        
        Returns:
            Dict[str, Any]: 에이전트 응답 (output 키에 최종 답변 포함)
        """
        if self.agent_executor is None:
            raise ValueError("에이전트가 초기화되지 않았습니다.")
        
        # 메시지 형식 변환
        messages = [{"role": "user", "content": input_text}]
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict):
                    messages.insert(-1, msg)
                elif hasattr(msg, "content"):
                    role = "user" if getattr(msg, "type", None) == "human" else "assistant"
                    messages.insert(-1, {"role": role, "content": str(msg.content)})
        
        # 에이전트 실행
        result = self.agent_executor.invoke({"messages": messages})
        
        # 결과 파싱
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last_msg = result["messages"][-1]
            content = getattr(last_msg, "content", "") if hasattr(last_msg, "content") else last_msg.get("content", "")
            if isinstance(content, list):
                text_items = [item.get("text", str(item)) for item in content if isinstance(item, dict) and item.get("type") == "text"]
                content = " ".join(text_items) if text_items else str(content)
            return {"output": str(content)}
        
        return {"output": str(result)}
    
    def update_system_message(self, new_message: str) -> None:
        """
        시스템 메시지 업데이트
        
        Args:
            new_message: 새로운 시스템 메시지
        """
        self.system_message = new_message
        self._build_agent()
    
    def get_agent_executor(self):
        """
        AgentExecutor 인스턴스 반환
        
        Returns:
            에이전트 실행기 인스턴스
        """
        return self.agent_executor

@wrap_tool_call
def handle_tool_errors(request, handler):
    """도구 실행 오류 처리"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"도구 오류: 입력을 확인하고 다시 시도하세요. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )