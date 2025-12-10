"""
Nornir와 NAPALM을 이용한 네트워크 장비 제어(MCP) 모듈 (모의 구현)
"""
from typing import Dict, Any

# from nornir import InitNornir
# from nornir_napalm.plugins.tasks import napalm_get

class RoutingMCP:
    """
    네트워크 장비의 라우팅 정보를 조회하는 클래스
    """
    def __init__(self, config_file: str = "config.yaml"):
        """
        실제 환경에서는 Nornir 초기화가 필요합니다.
        self.nr = InitNornir(config_file=config_file)
        """
        # 모의 구현을 위한 가상 라우팅 테이블
        self.mock_routing_table = {
            "192.168.1.1": { # gimpo-dmz-fw
                "100.130.23.45": "10.1.1.2" # -> core-switch-1
            },
            "192.168.1.10": { # core-switch-1
                "100.130.23.45": "10.0.0.6" # -> yeouido-dev-l3
            },
            "192.168.1.2": { # yeouido-dev-l3
                "100.130.23.45": "DIRECT" # Directly connected
            }
        }
        print("RoutingMCP가 모의(mock) 모드로 초기화되었습니다.")

    def get_next_hop(self, management_ip: str, destination_ip: str) -> Dict[str, Any]:
        """
        장비에서 목적지 IP로의 Next Hop 정보를 조회합니다.

        실제 구현 예시:
        nr_with_host = self.nr.filter(management_ip=management_ip)
        result = nr_with_host.run(
            task=napalm_get, getters=["get_route_to"], destination=destination_ip
        )
        # 결과 파싱 로직...
        """
        print(f"[MCP] {management_ip}에서 {destination_ip}로의 Next Hop 조회 시도...")
        
        # 모의 구현 로직
        device_routes = self.mock_routing_table.get(management_ip)
        if not device_routes:
            return {}
        
        next_hop_ip = device_routes.get(destination_ip)
        if not next_hop_ip:
            return {}

        print(f"[MCP] 조회 결과: Next Hop은 {next_hop_ip} 입니다.")
        return {"next_hop_ip": next_hop_ip}