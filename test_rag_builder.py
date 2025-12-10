import unittest
import os
from unittest.mock import MagicMock

from data.rag_builder import create_rag_retriever
from langchain_core.vectorstores import VectorStoreRetriever

class TestRagBuilder(unittest.TestCase):

    def setUp(self):
        """테스트를 위한 임시 JSON 파일을 생성합니다."""
        self.test_file = "test_data.json"
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write('[{"key": "value1"}, {"key": "value2"}]')

    def tearDown(self):
        """테스트 후 임시 파일을 삭제합니다."""
        os.remove(self.test_file)

    def test_create_rag_retriever(self):
        """RAG Retriever가 정상적으로 생성되는지 테스트합니다."""
        # 임베딩 모델을 모의(mock) 처리합니다.
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        retriever = create_rag_retriever(self.test_file, mock_embeddings)
        self.assertIsInstance(retriever, VectorStoreRetriever)

if __name__ == '__main__':
    unittest.main()
