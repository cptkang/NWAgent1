"""
예제 실행 파일
RAG와 Tools의 기본 사용법을 보여줍니다.
"""

import os
from ollama_client import LLMAPIClient
from rag import RAG
from tools import Tools
from agent import Agent


def example_rag_only():
    """RAG만 사용하는 예제"""
    print("=" * 60)
    print("예제 1: RAG만 사용 (Excel 파일 인덱싱)")
    print("=" * 60)
    
    # LLM API 클라이언트 초기화
    # API Gateway를 통해 Ollama에 접근 (API Key 인증)
    ollama_client = LLMAPIClient(
        api_endpoint="http://localhost:8000/api/chat",  # Gateway URL
        base_url="http://localhost:8000",
        chat_model="llama3.1",
        embedding_model="mxbai-embed-large",
        api_key=os.getenv("OLLAMA_API_KEY", "my-very-secret-key-123")  # API Key
    )
    
    # RAG 초기화
    rag = RAG(ollama_client=ollama_client)
    
    # Excel 파일 인덱싱 (예제 경로)
    excel_path = "./data/example.xlsx"  # 실제 Excel 파일 경로로 변경 필요
    try:
        rag.build_index(excel_path, persist_directory="./vector_index")
        
        # 검색 테스트
        results = rag.search("검색어", k=3)
        print(f"검색 결과: {len(results)}개 문서")
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc.page_content[:100]}...")
    except FileNotFoundError:
        print(f"Excel 파일을 찾을 수 없습니다: {excel_path}")


def example_tools_only():
    """Tools만 사용하는 예제"""
    print("\n" + "=" * 60)
    print("예제 2: IP 주소 분석 도구만 사용")
    print("=" * 60)
    
    from tools import ip_address_analyzer
    
    # IP 주소 분석 테스트
    test_ips = [
        "192.168.1.100/24",
        "10.0.0.50/16",
        "172.16.0.1/20"
    ]
    
    for ip in test_ips:
        print(f"\n입력: {ip}")
        result = ip_address_analyzer.invoke({"ip_address": ip})
        print(result)


def example_agent():
    """전체 에이전트 사용 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 전체 에이전트 사용 (RAG + Tools)")
    print("=" * 60)
    
    # LLM API 클라이언트 초기화
    # API Gateway를 통해 Ollama에 접근 (API Key 인증)
    ollama_client = LLMAPIClient(
        api_endpoint="http://localhost:8000/api/chat",  # Gateway URL
        base_url="http://localhost:8000",
        chat_model="llama3.1",
        embedding_model="mxbai-embed-large",
        api_key=os.getenv("OLLAMA_API_KEY", "my-very-secret-key-123")  # API Key
    )
    
    # RAG 초기화 (Excel 파일이 있는 경우)
    rag = None
    excel_path = "./data/example.xlsx"
    if os.path.exists(excel_path):
        rag = RAG(ollama_client=ollama_client)
        rag.build_index(excel_path, persist_directory="./vector_index")
    
    # Tools 초기화
    retriever = rag.get_retriever() if rag else None
    tools = Tools(retriever=retriever)
    
    # Agent 초기화
    agent = Agent(
        ollama_client=ollama_client,
        tools=tools,
        verbose=True
    )
    
    # 테스트 쿼리
    test_queries = [
        #"172.168.1.100/24의 네트워크 정보를 알려줘",
        #f"Excel 파일에서 10.136.59.59 의 [회사명]과 [IP 주소 대역]을 알려줘. 답변예시: [회사명], [IP 주소 대역]",
        #"10.136.0.50 주소가 10.136.0.0/16 대역에 포함되는지 확인해줘",
        "100.116.64.100의 주소를 사용하는 회사 정보를 알려줘."
        #"190.0.0.50/16 주소의 서브넷 마스크는 무엇인가요?"
    ]
    
    #if rag:
    #    test_queries.append("Excel 파일에서 172.168.1.1 의 클래스 정보를 찾아줘. 답변예시는 다음과 같다. [답변예시] 문의하신 [ip]정보의 클래스 [클래스]입니다. 답변은 항상 답변예시와 같은 형식으로 답변해줘.")
    
    for query in test_queries:
        print(f"\n질문: {query}")
        print("-" * 60)
        response = agent.invoke(query, chat_history=[])
        #test_queries.append(response['output'] + )
        print(f"답변: {response['output']}\n")


if __name__ == "__main__":
    print("Ollama 기반 RAG와 Tools 예제\n")
    
    # 예제 1: RAG만 사용
    # example_rag_only()
    
    # 예제 2: Tools만 사용
    #example_tools_only()
    
    # 예제 3: 전체 에이전트 사용
    example_agent()

