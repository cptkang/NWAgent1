"""
Tools 클래스
에이전트가 사용할 도구들을 정의합니다.
"""

import ipaddress
from typing import Optional
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool


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

