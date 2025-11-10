"""
메인 실행 파일
RAG와 Tools를 사용하는 Ollama 기반 에이전트 시스템
"""

import os
import argparse
from pathlib import Path

from ollama_client import OllamaClient
from rag import RAG
from tools import Tools
from agent import Agent


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Ollama 기반 RAG와 Tools 에이전트")
    parser.add_argument(
        "--excel",
        type=str,
        help="Excel 파일 경로 (RAG 인덱싱용)",
        default=None
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        help="벡터 인덱스 저장 디렉토리",
        default="./vector_index"
    )
    parser.add_argument(
        "--load-index",
        action="store_true",
        help="기존 인덱스 로드 (Excel 파일 로드 대신)"
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        help="채팅용 Ollama 모델 (tools 지원: llama3.1, llama3.2 등)",
        default="llama3.1"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="임베딩용 Ollama 모델",
        default="mxbai-embed-large"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Ollama 서버 URL",
        default="http://localhost:11434"
    )
    
    args = parser.parse_args()
    
    # 1. Ollama 클라이언트 초기화
    print("=" * 60)
    print("Ollama 클라이언트 초기화 중...")
    print("=" * 60)
    ollama_client = OllamaClient(
        base_url=args.base_url,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model
    )
    print(f"Chat Model: {args.chat_model}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Base URL: {args.base_url}\n")
    
    # 2. RAG 초기화 및 인덱싱
    print("=" * 60)
    print("RAG 시스템 초기화 중...")
    print("=" * 60)
    rag = RAG(ollama_client=ollama_client)
    
    if args.load_index:
        # 기존 인덱스 로드
        if os.path.exists(args.index_dir):
            rag.load_index(args.index_dir)
        else:
            print(f"오류: 인덱스 디렉토리를 찾을 수 없습니다: {args.index_dir}")
            return
    elif args.excel:
        # Excel 파일에서 인덱스 생성
        if not os.path.exists(args.excel):
            print(f"오류: Excel 파일을 찾을 수 없습니다: {args.excel}")
            return
        rag.build_index(
            excel_path=args.excel,
            persist_directory=args.index_dir
        )
    else:
        print("경고: Excel 파일 또는 인덱스 로드 옵션이 지정되지 않았습니다.")
        print("RAG 기능 없이 IP 분석 도구만 사용할 수 있습니다.\n")
        rag = None
    
    # 3. Tools 초기화
    print("\n" + "=" * 60)
    print("Tools 초기화 중...")
    print("=" * 60)
    retriever = rag.get_retriever() if rag else None
    tools = Tools(retriever=retriever)
    print(f"사용 가능한 도구: {[t.name for t in tools.get_tools()]}\n")
    
    # 4. Agent 초기화
    print("=" * 60)
    print("Agent 초기화 중...")
    print("=" * 60)
    agent = Agent(
        ollama_client=ollama_client,
        tools=tools,
        #system_message="You are a helpful assistant that can search Excel documents and analyze IP addresses.",
        verbose=True
    )
    
    
    print("\n")
    
    # 5. 대화형 실행
    print("=" * 60)
    print("에이전트 준비 완료! 질문을 입력하세요. (종료: 'quit' 또는 'exit')")
    print("=" * 60)
    print()
    
    chat_history = []
    
    while True:
        try:
            #user_input = input("사용자: ").strip()
            user_input = "172.168.1.100의 네트워크 정보를 알려줘."
            if user_input.lower() in ["quit", "exit", "종료"]:
                print("대화를 종료합니다.")
                break
            
            if not user_input:
                continue
            
            print("\n" + "-" * 60)
            response = agent.invoke(user_input, chat_history)
            print("-" * 60)
            print(f"\n에이전트: {response['output']}\n")
            
            # 대화 기록 업데이트 (간단한 버전)
            # 실제 구현에서는 LangChain의 메시지 형식을 사용해야 할 수 있음
            
        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {str(e)}\n")


if __name__ == "__main__":
    main()

