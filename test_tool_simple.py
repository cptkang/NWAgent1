"""
간단한 Tool 호출 테스트 스크립트
subnet_calculator tool이 호출되는지 빠르게 확인합니다.
LLMAPIClient를 사용하여 Gateway를 통해 Ollama에 접근합니다.
"""

import os
from ollama_client import LLMAPIClient
from langchain_core.messages import HumanMessage, ToolMessage
from tools import subnet_calculator


def simple_test():
    """간단한 tool 호출 테스트"""
    
    print("=" * 60)
    print("간단한 Tool 호출 테스트 (LLMAPIClient 사용)")
    print("=" * 60)
    
    # LLMAPIClient 초기화 (Gateway를 통해 Ollama에 접근)
    print("\n[1] LLMAPIClient 초기화 중...")
    ollama_client = LLMAPIClient(
        api_endpoint="http://localhost:8000/api/chat",  # Gateway URL
        base_url="http://localhost:8000",
        chat_model="llama3.1",
        embedding_model="mxbai-embed-large",
        api_key=os.getenv("OLLAMA_API_KEY", "my-very-secret-key-123")  # API Key
    )
    
    # LLM 인스턴스 가져오기
    llm = ollama_client.get_chat_llm()
    
    print("[2] Tool 바인딩 중...")
    # Tool 바인딩
    llm_with_tools = llm.bind_tools([subnet_calculator])
    
    # 테스트 쿼리
    query = "100.116.64.100의 24비트, 23비트, 22비트 서브넷 주소와 관련된 정보를 알려줘."
    
    print(f"\n질의: {query}\n")
    print("-" * 60)
    
    # 메시지 초기화
    messages = [HumanMessage(content=query)]
    
    # 최대 3회 반복
    max_iterations = 3
    tool_calls_processed = set()  # 처리된 tool call ID 추적
    
    for i in range(max_iterations):
        print(f"\n[단계 {i+1}] LLM 호출 중...")
        
        # LLM 호출
        response = llm_with_tools.invoke(messages)
        
        # 응답 출력
        if response.content:
            print(f"응답: {response.content[:300]}...")
        
        # Tool 호출 확인
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\n✅ Tool 호출 감지! ({len(response.tool_calls)}개)")
            
            # 이미 처리된 tool call이 있는지 확인
            new_tool_calls = []
            for tool_call in response.tool_calls:
                tool_id = tool_call.get("id", "")
                if tool_id not in tool_calls_processed:
                    new_tool_calls.append(tool_call)
                    tool_calls_processed.add(tool_id)
            
            if not new_tool_calls:
                print("\n⚠️ 모든 tool call이 이미 처리되었습니다. 최종 응답 생성 중...")
                # Tool 없이 최종 응답 생성
                messages.append(response)
                final_response = llm.invoke(messages)
                print(f"\n최종 응답: {final_response.content}")
                break
            
            for tool_call in new_tool_calls:
                name = tool_call.get("name", "")
                args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")
                
                print(f"\n  Tool: {name}")
                print(f"  Args: {args}")
                
                # Tool 실행
                if name == "subnet_calculator":
                    result = subnet_calculator.invoke(args)
                    print(f"  결과:\n{result}")
                    
                    # Tool 결과를 메시지에 추가
                    messages.append(response)
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id
                    ))
        else:
            print("\n✅ Tool 호출 없음 - 최종 응답")
            print(f"최종 응답: {response.content}")
            break
        
        # 마지막 반복이면 강제 종료
        if i == max_iterations - 1:
            print(f"\n⚠️ 최대 반복 횟수 도달. 최종 응답 생성 중...")
            # Tool 없이 최종 응답 생성
            messages.append(response)
            final_response = llm.invoke(messages)
            print(f"\n최종 응답: {final_response.content}")
            break
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        simple_test()
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        print("\n확인 사항:")
        print("  1. Gateway 서버 실행 중: python gateway.py")
        print("  2. Ollama 서버 실행 중: ollama serve")
        print("  3. llama3.1 모델 설치: ollama pull llama3.1")
        print("  4. 환경 변수 설정: export OLLAMA_API_KEY='your-api-key'")
        import traceback
        traceback.print_exc()

