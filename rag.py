"""
RAG (Retrieval-Augmented Generation) 클래스
Excel 파일을 읽어서 벡터화하고 검색 기능을 제공합니다.
"""

# 표준 라이브러리
import os
from typing import List, Optional

# 필수 의존성
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 로컬 모듈
from typing import Union
from ollama_client import OllamaClient, LLMAPIClient


class RAG:
    """Excel 파일 기반 RAG 시스템"""
    
    def __init__(
        self,
        ollama_client: Union[OllamaClient, LLMAPIClient],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        RAG 시스템 초기화
        
        Args:
            ollama_client: OllamaClient 또는 LLMAPIClient 인스턴스
            chunk_size: 문서 분할 시 청크 크기
            chunk_overlap: 문서 분할 시 청크 간 겹치는 부분
        """
        self.ollama_client = ollama_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore: Optional[FAISS] = None
        self.retriever: Optional[BaseRetriever] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_excel(self, file_path: str, mode: str = "single") -> List[Document]:
        """
        Excel 파일을 로드하여 Document 리스트로 변환
        
        Args:
            file_path: Excel 파일 경로
            mode: 파티셔닝 모드 ("single" 또는 "elements", 기본값: "single")
        
        Returns:
            List[Document]: 로드된 문서 리스트
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel 파일을 찾을 수 없습니다: {file_path}")
        
        loader = UnstructuredExcelLoader(
            file_path=file_path,
            mode=mode
        )
        docs = loader.load()
        return docs
    
    def build_index(
        self,
        excel_path: str,
        mode: str = "single",
        persist_directory: Optional[str] = None
    ) -> None:
        """
        Excel 파일을 벡터화하여 인덱스 생성
        
        Args:
            excel_path: Excel 파일 경로
            mode: 파티셔닝 모드 ("single" 또는 "elements", 기본값: "single")
            persist_directory: 인덱스 저장 디렉토리 (None이면 메모리에만 저장)
        """
        print(f"Excel 파일 로드 중: {excel_path}")
        docs = self.load_excel(excel_path, mode)
        
        print(f"로드된 문서 수: {len(docs)}")
        print("문서 분할 중...")
        splits = self.text_splitter.split_documents(docs)
        print(f"분할된 청크 수: {len(splits)}")
        
        print("임베딩 생성 중...")
        embeddings = self.ollama_client.get_embeddings()
        
        print("벡터 스토어 생성 중...")
        if persist_directory and os.path.exists(persist_directory):
            # 기존 인덱스 로드
            try:
                self.vectorstore = FAISS.load_local(
                    persist_directory,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                # 새 문서 추가
                self.vectorstore.add_documents(splits)
                print(f"기존 인덱스에 문서 추가 완료")
            except Exception as e:
                print(f"기존 인덱스 로드 실패, 새로 생성: {e}")
                self.vectorstore = FAISS.from_documents(
                    documents=splits,
                    embedding=embeddings
                )
        else:
            # 새 인덱스 생성
            self.vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=embeddings
            )
        
        # Retriever 생성
        self.retriever = self.vectorstore.as_retriever()
        print("벡터 스토어 생성 완료")
        
        # 인덱스 저장
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.vectorstore.save_local(persist_directory)
            print(f"인덱스 저장 완료: {persist_directory}")
    
    def load_index(self, persist_directory: str) -> None:
        """
        저장된 인덱스 로드
        
        Args:
            persist_directory: 인덱스가 저장된 디렉토리
        """
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"인덱스 디렉토리를 찾을 수 없습니다: {persist_directory}")
        
        embeddings = self.ollama_client.get_embeddings()
        self.vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()
        print(f"인덱스 로드 완료: {persist_directory}")
    
    def get_retriever(self) -> BaseRetriever:
        """
        Retriever 인스턴스 반환
        
        Returns:
            BaseRetriever: Retriever 인스턴스
        """
        if self.retriever is None:
            raise ValueError("인덱스가 아직 생성되지 않았습니다. build_index()를 먼저 호출하세요.")
        return self.retriever
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        검색 쿼리 실행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
        
        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if self.retriever is None:
            raise ValueError("인덱스가 아직 생성되지 않았습니다. build_index()를 먼저 호출하세요.")
        
        self.retriever.search_kwargs = {"k": k}
        return self.retriever.invoke(query)

