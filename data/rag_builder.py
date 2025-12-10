"""
JSON 파일로부터 RAG(Retrieval-Augmented Generation)를 위한
VectorStoreRetriever를 생성하는 모듈
"""
import os
import json
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


def create_rag_retriever(
    file_path: str,
    embedding_model: Embeddings,
    persist_directory: str = None,
    **kwargs
) -> VectorStoreRetriever:
    """
    JSON 파일을 읽어 FAISS 벡터 저장소를 만들고 Retriever를 반환합니다.
    persist_directory가 지정되면, 해당 경로에 벡터 저장소를 저장하거나 로드합니다.

    Args:
        file_path: JSON 파일 경로
        embedding_model: 문서를 임베딩할 모델
        persist_directory: 벡터 저장소를 저장/로드할 디렉토리 경로 (선택사항)
        **kwargs: Retriever에 전달할 추가 인자 (e.g., search_type="mmr")

    Returns:
        VectorStoreRetriever: 생성된 검색기
    """
    if persist_directory and os.path.exists(os.path.join(persist_directory, "index.faiss")):
        print(f"'{persist_directory}'에서 기존 벡터 저장소를 로드합니다.")
        vector_store = FAISS.load_local(
            persist_directory, embedding_model, allow_dangerous_deserialization=True
        )
    else:
        print(f"'{file_path}'로부터 새로운 벡터 저장소를 생성합니다.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 각 JSON 객체를 하나의 Document로 변환
        documents = [Document(page_content=json.dumps(item, ensure_ascii=False)) for item in data]

        vector_store = FAISS.from_documents(documents, embedding_model)

        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            print(f"새로운 벡터 저장소를 '{persist_directory}'에 저장합니다.")
            vector_store.save_local(persist_directory)

    return vector_store.as_retriever(**kwargs)