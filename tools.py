"""
Tools 클래스
에이전트가 사용할 도구들을 정의합니다.
"""

import ipaddress
from typing import Optional
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever

# create_retriever_tool import 시도 (다양한 경로 지원)
try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    try:
        from langchain_community.tools.retriever import create_retriever_tool
    except ImportError:
        try:
            from langchain_core.tools.retriever import create_retriever_tool
        except ImportError:
            # 직접 구현 (LangChain의 기본 구조 사용)
            from langchain_core.tools import StructuredTool
            from typing import Any
            
            # pydantic import 시도 (다양한 경로 지원)
            try:
                from langchain_core.pydantic_v1 import BaseModel, Field
            except ImportError:
                try:
                    from pydantic import BaseModel, Field
                except ImportError:
                    # 최후의 수단: 간단한 래퍼 클래스
                    class BaseModel:
                        pass
                    class Field:
                        def __init__(self, **kwargs):
                            pass
            
            def create_retriever_tool(
                retriever: BaseRetriever,
                name: str,
                description: str
            ) -> Any:
                """Retriever를 Tool로 변환하는 함수"""
                
                # pydantic이 있는 경우에만 args_schema 사용
                try:
                    class RetrieverInput(BaseModel):
                        query: str = Field(description="검색할 쿼리")
                    
                    def retrieve_docs(query: str) -> str:
                        """문서 검색 실행"""
                        docs = retriever.invoke(query)
                        return "\n\n".join([doc.page_content for doc in docs])
                    
                    return StructuredTool.from_function(
                        func=retrieve_docs,
                        name=name,
                        description=description,
                        args_schema=RetrieverInput
                    )
                except (NameError, TypeError):
                    # pydantic이 없는 경우 args_schema 없이 생성
                    def retrieve_docs(query: str) -> str:
                        """문서 검색 실행"""
                        docs = retriever.invoke(query)
                        return "\n\n".join([doc.page_content for doc in docs])
                    
                    return StructuredTool.from_function(
                        func=retrieve_docs,
                        name=name,
                        description=description
                    )


class Tools:
    """에이전트가 사용할 도구들을 관리하는 클래스"""
    
    def __init__(self, retriever: Optional[BaseRetriever] = None):
        """
        Tools 초기화
        
        Args:
            retriever: RAG를 위한 Retriever 인스턴스 (선택사항)
        """
        self.retriever = retriever
        self.tools = []
        
        # RAG Retriever Tool 추가
        if retriever is not None:
            self.add_retriever_tool()
        
        # IP 주소 분석 Tool 추가
        self.tools.append(ip_address_analyzer)
    
    def add_retriever_tool(
        self,
        name: str = "excel_knowledge_search",
        description: str = None
    ) -> None:
        """
        IP 주소를 받아 IP 주소체계을 어떤 클래스인지 분석하여 결과를 반환한다. 
        
        Args:
            ip_address: 분석할 IP 주소 (예: "192.168.1.100" 또는 "192.168.1.100/24")
            name: IP 주소 분석 툴 이름
            description: IP 주소를 받아 IP의 Class를 분석하여 결과를 반환한다. 
                        예를 들어 192.168.1.100/24 를 받으면 192.168.1.0/24 의 Class C라고 분석하여 결과를 반환한다.
        
        
        Returns:
            str: IP 주소 분석 결과 (Class A, B, C, D, E)
    
        Examples:
           192.168.1.100/24 를 받으면 192.168.1.0/24 의 Class C 라고 반환한다. 
        
        """
        if self.retriever is None:
            raise ValueError("Retriever가 설정되지 않았습니다.")
        
        if description is None:
            description = (
                "Excel 파일에서 로드된 지식 기반에 대한 정보를 검색할 때 사용합니다. "
                "Excel 파일의 내용에 대한 질문에 유용합니다."
            )
        
        retriever_tool = create_retriever_tool(
            self.retriever,
            name,
            description
        )
        
        # 기존 retriever tool이 있으면 제거하고 새로 추가
        self.tools = [t for t in self.tools if t.name != name]
        self.tools.insert(0, retriever_tool)
    
    def get_tools(self):
        """
        모든 도구 리스트 반환
        
        Returns:
            list: 도구 리스트
        """
        return self.tools


@tool
def ip_address_analyzer(ip_address: str, subnet_mask: Optional[str] = None) -> str:
    """
    IPv4 주소를 분석하여 네트워크 주소, 호스트 주소, 서브넷 마스크, 
    전체 IP 주소 대역을 계산합니다.
    
    Args:
        ip_address: 분석할 IP 주소 (예: "192.168.1.100" 또는 "192.168.1.100/24")
        subnet_mask: 서브넷 마스크 (CIDR 표기법이 포함된 경우 생략 가능, 예: "255.255.255.0")
    
    Returns:
        str: IP 주소 분석 결과 (네트워크 주소, 호스트 주소, 서브넷 마스크, IP 대역)
    
    Examples:
        ip_address_analyzer("192.168.1.100/24")
        ip_address_analyzer("192.168.1.100", "255.255.255.0")
    """
    try:
        # CIDR 표기법이 포함된 경우
        if "/" in ip_address:
            ip_network = ipaddress.IPv4Network(ip_address, strict=False)
            # 원본 IP 주소 추출 (CIDR 제거)
            original_ip = ip_address.split("/")[0]
        else:
            # IP 주소만 있는 경우
            original_ip = ip_address
            ip_addr = ipaddress.IPv4Address(ip_address)
            
            # 서브넷 마스크가 제공된 경우
            if subnet_mask:
                if "/" in subnet_mask:
                    # CIDR 표기법 (예: "/24")
                    prefix = int(subnet_mask.replace("/", ""))
                    ip_network = ipaddress.IPv4Network(
                        f"{ip_addr}/{prefix}",
                        strict=False
                    )
                else:
                    # 서브넷 마스크 주소 (예: "255.255.255.0")
                    mask_octets = subnet_mask.split(".")
                    prefix = sum(bin(int(x)).count("1") for x in mask_octets)
                    ip_network = ipaddress.IPv4Network(
                        f"{ip_addr}/{prefix}",
                        strict=False
                    )
            else:
                # 기본 서브넷 마스크 사용 (클래스 A, B, C 자동 감지)
                first_octet = int(str(ip_addr).split(".")[0])
                if first_octet < 128:
                    prefix = 8  # Class A
                elif first_octet < 192:
                    prefix = 16  # Class B
                else:
                    prefix = 24  # Class C
                ip_network = ipaddress.IPv4Network(
                    f"{ip_addr}/{prefix}",
                    strict=False
                )
        
        # 네트워크 정보 추출
        network_address = str(ip_network.network_address)
        broadcast_address = str(ip_network.broadcast_address)
        subnet_mask_str = str(ip_network.netmask)
        prefix_length = ip_network.prefixlen
        num_hosts = ip_network.num_addresses
        usable_hosts = num_hosts - 2  # 네트워크 주소와 브로드캐스트 주소 제외
        
        # 호스트 주소는 원본 IP 주소
        host_ip = original_ip
        
        result = f"""
IP 주소 분석 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력 IP 주소: {host_ip}
네트워크 주소: {network_address}
브로드캐스트 주소: {broadcast_address}
서브넷 마스크: {subnet_mask_str} (/{prefix_length})
전체 IP 주소 대역: {network_address} ~ {broadcast_address}
사용 가능한 호스트 수: {usable_hosts}개
전체 주소 수: {num_hosts}개
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """.strip()
        
        return result
    
    except ValueError as e:
        return f"오류: 잘못된 IP 주소 형식입니다. {str(e)}"
    except Exception as e:
        return f"오류: IP 주소 분석 중 오류가 발생했습니다. {str(e)}"

