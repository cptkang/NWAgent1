"""
네트워크 라우팅 분석 에이전트 실행 파일
"""
import os
import json
from dotenv import load_dotenv

from ollama_client import CustomOllamaChat, AuthenticatedOllamaEmbeddings
from data.rag_builder import create_rag_retriever
from tools.network_tools import FindSubnetInfoTool, FindDeviceInfoTool, GetNextHopTool
from mcp import RoutingMCP
from data.routing_agent_graph import RoutingAgentGraph, AgentState
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llm_creator import get_model_creator
from langchain_core.vectorstores import VectorStoreRetriever


def initialize_retrievers(embeddings: Embeddings) -> VectorStoreRetriever:
    """
    장비 정보(device)에 대한 RAG 검색기(Retriever)를 초기화합니다.
    벡터 저장소가 디스크에 존재하면 로드하고, 없으면 생성 후 저장합니다.
    """
    vector_store_path = "vector_store"
    device_index_path = os.path.join(vector_store_path, "devices")

    print("\n--- RAG Retriever 초기화 ---")
    device_retriever = create_rag_retriever(
        "data/network_devices.json",
        embeddings,
        persist_directory=device_index_path
    )
    print("--- RAG Retriever 초기화 완료 ---\n")
    
    return device_retriever

def initialize_models() -> tuple[dict[str, BaseChatModel], dict[str, Embeddings]]:
    """
    환경 변수에 따라 사용 가능한 모든 LLM 및 임베딩 모델을 초기화합니다.
    """
    llms = {}
    embeddings_map = {}

    print("\n--- LLM 및 임베딩 모델 초기화 시작 ---")

    # 지원하는 프로바이더 목록
    supported_providers = ["fabrix", "ollama"]

    for provider in supported_providers:
        creator = get_model_creator(provider)
        if creator:
            llm = creator.create_llm()
            if llm:
                llms[provider] = llm

            embeddings = creator.create_embeddings()
            if embeddings:
                embeddings_map[provider] = embeddings

    if not llms:
        raise ValueError("초기화할 수 있는 LLM 설정이 없습니다. .env 파일을 확인하세요.")

    print("--- LLM 및 임베딩 모델 초기화 완료 ---\n")
    return llms, embeddings_map

def main():
    """메인 실행 함수"""
    load_dotenv()

    # 1. LLM 및 임베딩 모델 초기화
    llms, embeddings_map = initialize_models()

    # 2. RAG 검색기(Retriever) 생성
    # RAG에 사용할 임베딩 모델 선택 (기본: ollama)
    rag_embedding_provider = os.getenv("RAG_EMBEDDING_PROVIDER", "ollama").lower()
    rag_embeddings = embeddings_map.get(rag_embedding_provider)
    if not rag_embeddings:
        # 설정된 프로바이더가 없으면 사용 가능한 첫 번째 프로바이더로 대체
        if not embeddings_map:
            raise ValueError("RAG를 위한 임베딩 모델을 찾을 수 없습니다.")
        fallback_provider = list(embeddings_map.keys())[0]
        print(f"경고: RAG 임베딩 프로바이더 '{rag_embedding_provider}'를 찾을 수 없어 '{fallback_provider}'로 대체합니다.")
        rag_embeddings = embeddings_map[fallback_provider]

    device_retriever = initialize_retrievers(rag_embeddings)

    # 3. MCP 및 도구 초기화
    # IP 서브넷 정보는 RAG 대신 JSON 파일을 직접 읽어 사용 (정확성 및 성능 향상)
    subnet_data_path = "data/ip_subnets.json"
    try:
        with open(subnet_data_path, 'r', encoding='utf-8') as f:
            subnet_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"'{subnet_data_path}' 파일을 로드할 수 없습니다: {e}")

    mcp_client = RoutingMCP()
    tools = [
        FindSubnetInfoTool(subnet_data=subnet_data),
        FindDeviceInfoTool(device_info_retriever=device_retriever),
        GetNextHopTool(mcp=mcp_client)
    ]

    # 4. LangGraph 에이전트 생성
    agent_graph = RoutingAgentGraph(llms, tools).graph

    # 5. 사용자 입력 및 에이전트 실행
    print("네트워크 라우팅 분석 에이전트입니다. 'exit'를 입력하면 종료됩니다.")
    while True:
        try:
            #user_input = input("질문: ")
            user_input = "172.168.50.40이 100.150.50.40과 통신해야 된다. 라우팅 정보가 설정되어 있는지 확인해줘. "
            if user_input.lower() == 'exit':
                break

            initial_state: AgentState = {
                "user_prompt": user_input,
                "source_ip": "",
                "destination_ip": "",
                "source_location": None,
                "destination_location": None,
                "environment": None,
                "trace_path": [],
                "current_hop_ip": "",
                "final_answer": "",
                "error_message": ""
            }

            final_state = agent_graph.invoke(initial_state)
            print("\n--- 최종 답변 ---")
            print(final_state.get("final_answer", "답변을 생성하지 못했습니다."))
            print("-" * 17 + "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
