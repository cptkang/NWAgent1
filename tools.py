"""
Tools 클래스
에이전트가 사용할 도구들을 정의합니다.
"""

import ipaddress
import re
from typing import Optional
from langchain_core.tools import Tool, tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field


class RAGIpRangeValidationInput(BaseModel):
    """RAG 기반 IP 대역 검증 도구에 사용할 입력 스키마"""
    
    query: str = Field(
        ...,
        description="RAG 검색에 사용할 키워드 (회사명, 서비스명, 장비명, 설명 등)"
    )
    ip_address: str = Field(
        ...,
        description="검증할 IPv4 주소 (예: 10.136.59.59)"
    )
    expected_range: Optional[str] = Field(
        default=None,
        description="사용자가 확인하고 싶은 IP 대역 (CIDR, 예: 10.136.0.0/19)"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="RAG에서 확인할 문서 수 (1-10)"
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
            self.tools.append(self._create_rag_ip_range_tool())
        
        # IP 주소 분석 Tool 추가
        self.tools.append(ip_address_analyzer)
        
        # IP 대역 포함 확인 Tool 추가
        self.tools.append(ip_range_checker)
        
        # IP 주소 대역 계산 Tool 추가
        self.tools.append(ip_range_calculator)
    
    def add_retriever_tool(
        self,
        name: str = "excel_knowledge_search",
        description: str = None
    ) -> None:
        """
        RAG Retriever를 Tool로 추가합니다.
        
        Args:
            name: Retriever Tool 이름
            description: Retriever Tool 설명
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
    
    def _create_rag_ip_range_tool(self) -> Tool:
        """
        RAG 결과를 기반으로 IP 대역 포함 여부를 확인하는 LangChain Tool을 생성합니다.
        """
        if self.retriever is None:
            raise ValueError("RAG Retriever가 설정되지 않았습니다.")
        
        def _rag_ip_range_checker(
            query: str,
            ip_address: str,
            expected_range: Optional[str] = None,
            top_k: int = 4
        ) -> str:
            return self._validate_ip_range_with_rag(
                query=query,
                ip_address=ip_address,
                expected_range=expected_range,
                top_k=top_k
            )
        
        return Tool.from_function(
            name="rag_ip_range_verifier",
            description=(
                "Excel에서 구축한 RAG 지식 기반을 검색해 특정 IPv4 주소가 어떤 IP 대역에 속하는지 확인합니다. "
                "회사명이나 장비명 같은 키워드를 query로 제공해주면 더 정확합니다."
            ),
            func=_rag_ip_range_checker,
            args_schema=RAGIpRangeValidationInput
        )
    
    def _validate_ip_range_with_rag(
        self,
        query: str,
        ip_address: str,
        expected_range: Optional[str],
        top_k: int
    ) -> str:
        """RAG 검색 결과와 IP 연산을 결합해 IP 대역을 검증합니다."""
        if self.retriever is None:
            return "오류: RAG Retriever가 활성화되지 않았습니다. 먼저 Excel 인덱스를 생성하세요."
        
        try:
            target_ip = ipaddress.IPv4Address(ip_address)
        except ValueError as exc:
            return f"오류: 잘못된 IP 주소 형식입니다. ({exc})"
        
        expected_membership = None
        expected_network = None
        if expected_range:
            try:
                expected_network = ipaddress.IPv4Network(expected_range, strict=False)
                expected_membership = target_ip in expected_network
            except ValueError as exc:
                return f"오류: 잘못된 CIDR 표기법입니다. ({exc})"
        
        retriever_to_use = None
        if hasattr(self.retriever, "with_search_kwargs"):
            try:
                retriever_to_use = self.retriever.with_search_kwargs({"k": top_k})
            except Exception:
                retriever_to_use = None
        
        try:
            raw_docs = (
                retriever_to_use.invoke(query)
                if retriever_to_use is not None
                else self.retriever.invoke(query)
            )
        except Exception as exc:
            return f"오류: RAG 검색 중 문제가 발생했습니다. ({exc})"
        
        docs = raw_docs or []
        if not isinstance(docs, list):
            docs = [docs]
        docs = [doc for doc in docs if doc][:top_k]
        
        if not docs:
            return (
                "RAG 검색 결과가 없습니다. 다른 키워드(query)를 사용하거나 "
                "Excel 데이터에 해당 IP 정보가 포함되어 있는지 확인하세요."
            )
        
        range_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}/\d{1,2}\b")
        doc_summaries = []
        matching_ranges = []
        extracted_ranges = []
        
        for idx, doc in enumerate(docs, start=1):
            content = (getattr(doc, "page_content", "") or "").strip()
            ranges = list(dict.fromkeys(range_pattern.findall(content)))
            metadata = getattr(doc, "metadata", {}) or {}
            source = str(metadata.get("source", "알 수 없음"))
            doc_summaries.append({
                "index": idx,
                "source": source,
                "ranges": ranges or ["(대역 정보 없음)"]
            })
            
            for rng in ranges:
                extracted_ranges.append(rng)
                try:
                    network = ipaddress.IPv4Network(rng, strict=False)
                except ValueError:
                    continue
                
                if target_ip in network:
                    matching_ranges.append({
                        "range": rng,
                        "source": source,
                        "doc_index": idx
                    })
        
        header = [
            "RAG 기반 IP 대역 검증 결과:",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ]
        header.append(f"검색 키워드: {query}")
        header.append(f"검증 대상 IP 주소: {ip_address}")
        if expected_range:
            status_icon = "✅" if expected_membership else "⚠️"
            header.append(
                f"사용자 제공 대역: {expected_range} → {status_icon} "
                f"{'포함됨' if expected_membership else '포함되지 않음'}"
            )
        header.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if matching_ranges:
            conclusion = [
                "확인 결과:",
                f"✅ RAG 문서에서 {ip_address}를 포함하는 대역을 찾았습니다."
            ]
            for match in matching_ranges:
                conclusion.append(
                    f"  - {match['range']} (문서 {match['doc_index']}, 출처: {match['source']})"
                )
        elif extracted_ranges:
            conclusion = [
                "확인 결과:",
                f"⚠️ 검색된 문서 {len(docs)}건에서 총 {len(extracted_ranges)}개의 대역을 발견했지만 "
                f"{ip_address}를 포함하는 대역은 없습니다."
            ]
        else:
            conclusion = [
                "확인 결과:",
                "⚠️ 검색된 문서에서 IP 대역 정보를 추출하지 못했습니다. "
                "Excel 데이터에 CIDR 표기가 포함되어 있는지 확인하세요."
            ]
        
        doc_lines = ["", "검토한 문서 요약:"]
        for summary in doc_summaries:
            doc_lines.append(
                f"  - 문서 {summary['index']} ({summary['source']}): "
                f"{', '.join(summary['ranges'])}"
            )
        
        return "\n".join(header + conclusion + doc_lines)
    
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


@tool
def ip_range_checker(ip_range: str, ip_address: str) -> str:
    """
    IP 대역에 특정 32비트 IP 주소가 포함되는지 확인합니다.
    
    Args:
        ip_range: IP 대역 (CIDR 표기법, 예: "192.168.1.0/24" 또는 "10.0.0.0/16")
        ip_address: 확인할 32비트 IP 주소 (예: "192.168.1.100")
    
    Returns:
        str: IP 주소가 대역에 포함되는지 여부와 상세 정보
    
    Examples:
        ip_range_checker("192.168.1.0/24", "192.168.1.100")
        ip_range_checker("10.0.0.0/16", "10.0.1.50")
    """
    try:
        # IP 대역 파싱
        network = ipaddress.IPv4Network(ip_range, strict=False)
        
        # IP 주소 파싱
        ip = ipaddress.IPv4Address(ip_address)
        
        # 포함 여부 확인
        is_in_range = ip in network
        
        # 결과 생성
        network_address = str(network.network_address)
        broadcast_address = str(network.broadcast_address)
        subnet_mask = str(network.netmask)
        prefix_length = network.prefixlen
        
        if is_in_range:
            result = f"""
IP 대역 포함 확인 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
결과: ✅ 포함됨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
확인 IP 주소: {ip_address}
IP 대역: {ip_range}
네트워크 주소: {network_address}
브로드캐스트 주소: {broadcast_address}
서브넷 마스크: {subnet_mask} (/{prefix_length})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IP 주소 {ip_address}는 {ip_range} 대역에 포함됩니다.
            """.strip()
        else:
            result = f"""
IP 대역 포함 확인 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
결과: ❌ 포함되지 않음
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
확인 IP 주소: {ip_address}
IP 대역: {ip_range}
네트워크 주소: {network_address}
브로드캐스트 주소: {broadcast_address}
서브넷 마스크: {subnet_mask} (/{prefix_length})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IP 주소 {ip_address}는 {ip_range} 대역에 포함되지 않습니다.
            """.strip()
        
        return result
    
    except ValueError as e:
        return f"오류: 잘못된 IP 주소 또는 대역 형식입니다. {str(e)}"
    except Exception as e:
        return f"오류: IP 대역 확인 중 오류가 발생했습니다. {str(e)}"


@tool
def ip_range_calculator(ip_address: str, cidr_bits: int) -> str:
    """
    IP 주소와 CIDR 비트 수를 받아 해당 IP 주소 대역 정보를 계산합니다.
    
    Args:
        ip_address: IP 주소 (예: "192.168.1.100")
        cidr_bits: CIDR 비트 수 (0-32, 예: 24)
    
    Returns:
        str: IP 주소 대역 정보 (네트워크 주소, 브로드캐스트 주소, 사용 가능한 IP 범위 등)
    
    Examples:
        ip_range_calculator("192.168.1.100", 24)
        ip_range_calculator("10.0.0.50", 16)
    """
    try:
        # CIDR 비트 수 유효성 검사
        if not (0 <= cidr_bits <= 32):
            return f"오류: CIDR 비트 수는 0-32 사이의 값이어야 합니다. 입력값: {cidr_bits}"
        
        # IP 주소 파싱
        ip = ipaddress.IPv4Address(ip_address)
        
        # CIDR 표기법으로 네트워크 생성
        network = ipaddress.IPv4Network(f"{ip}/{cidr_bits}", strict=False)
        
        # 네트워크 정보 추출
        network_address = str(network.network_address)
        broadcast_address = str(network.broadcast_address)
        subnet_mask = str(network.netmask)
        prefix_length = network.prefixlen
        num_hosts = network.num_addresses
        usable_hosts = num_hosts - 2  # 네트워크 주소와 브로드캐스트 주소 제외
        
        # 첫 번째와 마지막 사용 가능한 호스트 주소
        if num_hosts > 2:
            first_host = str(network.network_address + 1)
            last_host = str(network.broadcast_address - 1)
            host_range = f"{first_host} ~ {last_host}"
        else:
            host_range = "사용 가능한 호스트 없음 (네트워크 주소와 브로드캐스트 주소만 존재)"
        
        # 결과 생성
        result = f"""
IP 주소 대역 계산 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력 IP 주소: {ip_address}
CIDR 비트 수: /{cidr_bits}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
네트워크 주소: {network_address}
브로드캐스트 주소: {broadcast_address}
서브넷 마스크: {subnet_mask} (/{prefix_length})
IP 대역: {network_address}/{prefix_length}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 주소 수: {num_hosts}개
사용 가능한 호스트 수: {usable_hosts}개
사용 가능한 호스트 범위: {host_range}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """.strip()
        
        return result
    
    except ValueError as e:
        return f"오류: 잘못된 IP 주소 형식입니다. {str(e)}"
    except Exception as e:
        return f"오류: IP 주소 대역 계산 중 오류가 발생했습니다. {str(e)}"
