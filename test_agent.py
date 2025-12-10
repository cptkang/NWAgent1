import os
from ollama_client import LLMAPIClient
from rag import RAG
from tools import Tools
from agent import Agent

def test_agent():
    """Tests the full agent functionality."""
    print("=" * 60)
    print("Testing agetnt")
    print("=" * 60)
    
    # 1. Initialize LLM Client
    ollama_client = LLMAPIClient(
        api_endpoint="http://localhost:8000/api/chat",
        base_url="http://localhost:8000",
        chat_model="llama3.1",
        embedding_model="mxbai-embed-large",
        api_key=os.getenv("OLLAMA_API_KEY", "my-very-secret-key-123")
    )

    # 2. Initialize RAGs
    device_rag = RAG(ollama_client)
    device_rag.build_index_from_json(
        json_path="./data/network_devices.json",
        persist_directory="./vector_index_devices"
    )

    ip_info_rag = RAG(ollama_client)
    ip_info_rag.build_index_from_json(
        json_path="./data/ip_info.json",
        persist_directory="./vector_index_ip_info"
    )

    # 3. Initialize Tools
    tools = Tools()

    # 4. Initialize Agent
    agent = Agent(
        ollama_client=ollama_client,
        tools=tools,
        device_rag=device_rag,
        ip_info_rag=ip_info_rag,
        verbose=True
    )

    # 5. Invoke Agent with a test query
    query = "192.168.1.50이 10.130.23.100과 통신해야 된다. 라우팅 정보가 설정되어 있는지 확인해줘."
    response = agent.invoke(query, chat_history=[])
    print(f"Agent response:\n{response['output']}")
    
    query = "김포센터 DMZ에 있는 100.120.50.40의 서버가 여의도센터 개발환경의 내부망에 있는 100.130.23.45와 통신하려고 한다. 라우팅 설정이 있는지 확인해줘."
    response = agent.invoke(query, chat_history=[])
    print(f"Agent response:\n{response['output']}")


if __name__ == "__main__":
    test_agent()
