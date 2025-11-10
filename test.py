from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

# 1. Tool 정의
@tool
def get_current_weather(
    city: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    """특정 도시의 현재 날씨를 가져오는 데 유용한 함수."""
    if "서울" in city:
        return f"{city}의 현재 기온은 25도이며, 맑은 날씨다. (단위: {unit})"
    elif "뉴욕" in city:
        return f"{city}은 현재 흐리고 15도이다. (단위: {unit})"
    else:
        return f"{city}에 대한 현재 날씨 정보는 제공할 수 없다."

# 2. ChatOllama 모델 (툴 호출 지원 모델 사용)
base_llm = ChatOllama(
    model="llama3.1",  # 또는 tool calling에 튜닝된 gpt-oss:20b 등
    temperature=0,
)

# 3. Tool 바인딩
llm_with_tools = base_llm.bind_tools([get_current_weather])

# 4. 유저 입력
user_query = "내일 서울의 날씨는 어때? 온도는 섭씨로 알려줘. 반드시 get_current_weather 도구를 사용해서 알려줘."

# 5. 1차 호출: 모델이 tool call을 결정
ai_msg: AIMessage = llm_with_tools.invoke(user_query)

print("--- [Raw AIMessage] ---")
print(ai_msg)
print("\n--- [Tool Calls] ---")
print(ai_msg.tool_calls)

# 6. tool_calls를 실제로 실행
tool_answer = None
if ai_msg.tool_calls:
    for call in ai_msg.tool_calls:
        if call["name"] == "get_current_weather":
            args = call["args"]
            # @tool 로 만든 객체는 .invoke로 실행
            tool_answer = get_current_weather.invoke(args)
            print("\n✅ Tool 실행 결과:", tool_answer)

# 7. (선택) Tool 결과를 다시 모델에 넘겨서 자연어 답변 생성
if tool_answer is not None:
    messages = [
        HumanMessage(content=user_query),
        ai_msg,
        ToolMessage(
            content=tool_answer,
            name="get_current_weather",
            tool_call_id=ai_msg.tool_calls[0]["id"],
        ),
    ]
    final_msg: AIMessage = base_llm.invoke(messages)
    print("\n--- [최종 자연어 응답] ---")
    print(final_msg.content)
