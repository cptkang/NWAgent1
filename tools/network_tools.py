"""
네트워크 라우팅 분석을 위한 LangChain 도구(Tool) 정의
"""
import ipaddress
import json
from typing import Type, Dict, Any, List

from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import BaseModel, Field

from mcp import RoutingMCP


def ip_in_subnet(ip: str, subnet: str) -> bool:
    """IP 주소가 주어진 서브넷에 포함되는지 확인합니다."""
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(subnet)
    except ValueError:
        return False


class FindSubnetInfoInput(BaseModel):
    ip_address: str = Field(description="소속 정보를 찾을 32비트 IP 주소")

class FindSubnetInfoTool(BaseTool):
    """
    IP 주소가 어떤 서브넷에 속하는지 RAG를 통해 찾아내는 도구.
    서브넷 정보, 계열사, 위치, 장비명 등을 반환합니다.
    """
    name: str = "find_subnet_info"
    description: str = "IP 주소가 속한 서브넷, 계열사, 위치, 장비명 등의 정보를 찾습니다."
    args_schema: Type[BaseModel] = FindSubnetInfoInput
    # RAG(VectorStoreRetriever) 대신, 로드된 전체 서브넷 데이터를 직접 사용합니다.
    subnet_data: List[Dict[str, Any]]

    def _run(self, ip_address: str) -> Dict[str, Any]:
        """
        도구 실행 로직.
        메모리에 로드된 전체 서브넷 목록을 검색하여 가장 구체적인(longest prefix match) 정보를 반환합니다.
        이 방식은 벡터 검색보다 IP 주소 조회에 훨씬 더 정확하고 안정적입니다.
        """
        print(f"   - 전체 {len(self.subnet_data)}개 서브넷에서 {ip_address}에 대한 최적 정보 검색...")

        best_match = None
        longest_prefix = -1

        # 전체 서브넷 데이터를 순회하며 Longest Prefix Match 수행
        for subnet_info in self.subnet_data:
            try:
                subnet_cidr = subnet_info.get("subnet")
                if not subnet_cidr:
                    continue

                # IP 주소가 현재 서브넷에 포함되는지 확인
                if ip_in_subnet(ip_address, subnet_cidr):
                    network = ipaddress.ip_network(subnet_cidr)
                    # 현재까지 찾은 최적의 서브넷보다 더 구체적인지(prefix가 긴지) 확인
                    if network.prefixlen > longest_prefix:
                        longest_prefix = network.prefixlen
                        best_match = subnet_info
            except ValueError:
                # ipaddress 라이브러리에서 발생할 수 있는 오류 처리
                continue

        if best_match:
            print(f"   - 최적 서브넷 찾음: {best_match.get('subnet')}")
            return best_match

        return {"error": f"{ip_address}에 대한 서브넷 정보를 찾을 수 없습니다."}

class FindDeviceInfoInput(BaseModel):
    device_name: str = Field(description="정보를 찾을 네트워크 장비의 이름")

class FindDeviceInfoTool(BaseTool):
    """
    장비명을 기반으로 Management IP, Serial IP 등의 상세 정보를 RAG를 통해 찾아내는 도구.
    """
    name: str = "find_device_info"
    description: str = "네트워크 장비의 이름으로 Management IP, Serial IP 등의 상세 정보를 찾습니다."
    args_schema: Type[BaseModel] = FindDeviceInfoInput
    device_info_retriever: VectorStoreRetriever

    def _run(self, device_name: str) -> Dict[str, Any]:
        """도구 실행 로직"""
        # invoke()는 소수의 문서만 반환하므로, 정확한 장비명을 찾지 못할 수 있습니다.
        # vectorstore에서 직접 더 많은 후보(k=10)를 가져와서 필터링합니다.
        docs = self.device_info_retriever.vectorstore.similarity_search(device_name, k=10)
        for doc in docs:
            try:
                device_info = json.loads(doc.page_content)
                if device_info.get("device_name") == device_name:
                    return device_info
            except json.JSONDecodeError:
                continue
        return {"error": f"{device_name}에 대한 장비 정보를 찾을 수 없습니다."}


class GetNextHopInput(BaseModel):
    management_ip: str = Field(description="라우팅 정보를 조회할 장비의 Management IP")
    destination_ip: str = Field(description="최종 목적지 IP 주소")

class GetNextHopTool(BaseTool):
    """
    주어진 장비에서 특정 목적지로 가는 Next Hop 라우팅 정보를 조회하는 도구.
    MCP(Nornir/NAPALM)를 호출합니다.
    """
    name: str = "get_next_hop"
    description: str = "장비의 라우팅 테이블을 조회하여 목적지 IP에 대한 Next Hop 주소를 찾습니다."
    args_schema: Type[BaseModel] = GetNextHopInput
    mcp: RoutingMCP

    def _run(self, management_ip: str, destination_ip: str) -> Dict[str, Any]:
        """도구 실행 로직"""
        try:
            result = self.mcp.get_next_hop(
                management_ip=management_ip,
                destination_ip=destination_ip
            )
            if not result or not result.get("next_hop_ip"):
                return {"error": f"{management_ip} 장비에서 {destination_ip}로 가는 경로를 찾을 수 없습니다."}
            return result
        except Exception as e:
            return {"error": f"라우팅 정보 조회 중 오류 발생: {e}"}
