"""
subnet_calculator tool 호출 테스트 스크립트
LLM이 tool을 제대로 호출하는지 확인합니다.
LLMAPIClient를 사용하여 Gateway를 통해 Ollama에 접근합니다.
"""

import os
from ollama_client import LLMAPIClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import subnet_calculator
import json


def test_subnet_tool():
    """subnet_calculator tool 호출 테스트"""
    
    print("=" * 60)
    print("subnet_calculator Tool 호출 테스트 (LLMAPIClient 사용)")
    print("=" * 60)
    
    # 1. LLMAPIClient 초기화 (Gateway를 통해 Ollama에 접근)
    print("\n[1] LLMAPIClient 초기화 중...")
    ollama_client = LLMAPIClient(
        api_endpoint="http://localhost:8000/api/chat",  # Gateway URL
        base_url="http://localhost:8000",
        chat_model="llama3.1",
        embedding_model="mxbai-embed-large",
        api_key=os.getenv("OLLAMA_API_KEY", "my-very-secret-key-123")  # API Key
    )
    
    # LLM 인스턴스 가져오기
    base_llm = ollama_client.get_chat_llm()
    
    # 2. Tool 바인딩
    print("[2] Tool 바인딩 중...")
    print(f"   - 사용 가능한 tool: subnet_calculator")
    llm_with_tools = base_llm.bind_tools([subnet_calculator])
    
    # 3. 테스트 쿼리들
    test_queries = [
        "100.116.64.100의 24비트, 23비트, 22비트 서브넷 주소와 관련된 정보를 알려줘.",
        "100.116.64.100의 23비트 서브넷 주소를 알려줘.",
        "IP 주소 192.168.1.100의 24비트 서브넷 정보를 계산해줘.",
    ]
    
    for query_idx, user_query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"테스트 {query_idx}: {user_query}")
        print("=" * 60)
        
        # 4. 메시지 초기화
        messages = [HumanMessage(content=user_query)]
        
        # 5. Tool 호출 루프 (최대 5회 반복)
        max_iterations = 5
        tool_calls_processed = set()  # 처리된 tool call ID 추적
        
        for iteration in range(max_iterations):
            print(f"\n--- [반복 {iteration + 1}] ---")
            
            # 6. LLM 호출
            try:
                response: AIMessage = llm_with_tools.invoke(messages)
                print(f"\n[LLM 응답]")
                print(f"   Content: {response.content[:200] if response.content else '(없음)'}...")
                
                # 7. Tool 호출 확인
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"\n[✅ Tool 호출 감지!]")
                    print(f"   Tool Calls 수: {len(response.tool_calls)}")
                    
                    # 이미 처리된 tool call 필터링
                    new_tool_calls = []
                    for tool_call in response.tool_calls:
                        tool_id = tool_call.get("id", "")
                        if tool_id not in tool_calls_processed:
                            new_tool_calls.append(tool_call)
                            tool_calls_processed.add(tool_id)
                    
                    if not new_tool_calls:
                        print(f"\n[⚠️ 모든 tool call이 이미 처리되었습니다. 최종 응답 생성 중...]")
                        # Tool 없이 최종 응답 생성
                        messages.append(response)
                        final_response = base_llm.invoke(messages)
                        print(f"\n[최종 응답]")
                        print(f"   {final_response.content}")
                        break
                    
                    for tool_call_idx, tool_call in enumerate(new_tool_calls, 1):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_id = tool_call.get("id", "")
                        
                        print(f"\n   Tool Call #{tool_call_idx}:")
                        print(f"      Name: {tool_name}")
                        print(f"      Args: {json.dumps(tool_args, indent=8, ensure_ascii=False)}")
                        print(f"      ID: {tool_id}")
                        
                        # 8. Tool 실행
                        if tool_name == "subnet_calculator":
                            print(f"\n   [Tool 실행 중...]")
                            try:
                                tool_result = subnet_calculator.invoke(tool_args)
                                print(f"   [✅ Tool 실행 성공]")
                                print(f"   결과 (처음 200자): {tool_result[:200]}...")
                                
                                # Tool 결과를 메시지에 추가
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_id
                                )
                                messages.append(response)  # AI 응답 추가
                                messages.append(tool_message)  # Tool 결과 추가
                                
                            except Exception as e:
                                print(f"   [❌ Tool 실행 실패]: {str(e)}")
                                tool_message = ToolMessage(
                                    content=f"오류: {str(e)}",
                                    tool_call_id=tool_id
                                )
                                messages.append(response)
                                messages.append(tool_message)
                        else:
                            print(f"   [⚠️ 알 수 없는 tool: {tool_name}]")
                            tool_message = ToolMessage(
                                content=f"알 수 없는 도구: {tool_name}",
                                tool_call_id=tool_id
                            )
                            messages.append(response)
                            messages.append(tool_message)
                else:
                    # Tool 호출이 없으면 최종 응답
                    print(f"\n[최종 응답]")
                    print(f"   {response.content}")
                    break
                
                # 마지막 반복이면 강제 종료
                if iteration == max_iterations - 1:
                    print(f"\n[⚠️ 최대 반복 횟수 도달. 최종 응답 생성 중...]")
                    # Tool 없이 최종 응답 생성
                    messages.append(response)
                    final_response = base_llm.invoke(messages)
                    print(f"\n[최종 응답]")
                    print(f"   {final_response.content}")
                    break
                    
            except Exception as e:
                print(f"\n[❌ 오류 발생]: {str(e)}")
                import traceback
                traceback.print_exc()
                break
    
    print(f"\n{'=' * 60}")
    print("테스트 완료")
    print("=" * 60)


def test_tool_directly():
    """Tool을 직접 호출하여 작동 확인"""
    print("\n" + "=" * 60)
    print("Tool 직접 호출 테스트")
    print("=" * 60)
    
    test_cases = [
        {"ip_address": "100.116.64.100", "cidr_bits": 24},
        {"ip_address": "100.116.64.100", "cidr_bits": 23},
        {"ip_address": "100.116.64.100", "cidr_bits": 22},
    ]
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n[테스트 {idx}]")
        print(f"   입력: IP={test_case['ip_address']}, CIDR={test_case['cidr_bits']}비트")
        try:
            result = subnet_calculator.invoke(test_case)
            print(f"   결과:")
            print(result)
        except Exception as e:
            print(f"   [❌ 오류]: {str(e)}")


if __name__ == "__main__":
    import sys
    
    # 명령줄 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "direct":
        # Tool 직접 호출 테스트
        test_tool_directly()
    else:
        # LLM을 통한 Tool 호출 테스트
        try:
            test_subnet_tool()
        except Exception as e:
            print(f"\n[❌ 테스트 실패]: {str(e)}")
            print("\n[도움말]")
            print("  1. Gateway 서버가 실행 중인지 확인하세요: python gateway.py")
            print("  2. Ollama 서버가 실행 중인지 확인하세요: ollama serve")
            print("  3. llama3.1 모델이 설치되어 있는지 확인하세요: ollama list")
            print("  4. 환경 변수 설정: export OLLAMA_API_KEY='your-api-key'")
            print("  5. Tool 직접 호출 테스트: python test_subnet_tool.py direct")
            import traceback
            traceback.print_exc()

