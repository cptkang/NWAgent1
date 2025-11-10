from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import Literal

# 1. 사용할 Tool 정의 (Python 함수)
# @tool 데코레이터를 사용하여 LangChain Tool로 정의하고,
# Docstring과 Type Hint를 사용하여 모델에게 사용법을 명시한다.
@tool
def get_current_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """특정 도시의 현재 날씨를 가져오는 데 유용한 함수."""
    if "서울" in city:
        return f"{city}의 현재 기온은 25도이며, 맑은 날씨다. (단위: {unit})"
    elif "뉴욕" in city:
        return f"{city}은 현재 흐리고 15도이다. (단위: {unit})"
    else:
        return f"{city}에 대한 현재 날씨 정보는 제공할 수 없다."

# 2. ChatOllama 모델 인스턴스화
# Ollama 서버가 로컬에서 실행 중이어야 한다.
# model은 Ollama에 'llama3'와 같이 pull 되어 있는 모델이어야 한다.
base_llm = ChatOllama(model="llama3.1", temperature=0)

# 3. Tool Binding: 모델에 Tool의 존재와 스키마를 알려준다.
# bind_tools()를 통해 모델은 이 함수를 호출할 수 있는 능력을 얻는다.
llm_with_tools = base_llm.bind_tools([get_current_weather])

# 4. Tool 사용을 유도하는 프롬프트로 모델 호출 (Agent Loop의 첫 단계)
prompt = "내일 서울의 날씨는 어때? 온도는 섭씨로 알려줘."

# 모델 호출 및 결과 확인
response = llm_with_tools.invoke(prompt)

# 결과 출력
print("--- [모델 응답 객체 (Response)] ---")
print(type(response))
print("\n--- [모델 응답 Content] ---")
print(response.content)
print("\n--- [Tool Calls] ---")
print(response.tool_calls)