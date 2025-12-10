from typing import Dict, Any

# import os
# from nornir import InitNornir
# from nornir_napalm.plugins.tasks import napalm_get
# from nornir.core.exceptions import NornirExecutionError


class RoutingMCP:
    """
    네트워크 장비의 라우팅 정보를 조회하는 클래스 (모의 구현).
    기존에 data/nornir_mcp.py에 있던 mock 구현을 패키지 충돌 없이 여기로 옮겼습니다.
    """
    def __init__(self, config_file: str = "config.yaml"):
        """
        실제 환경에서는 Nornir 초기화가 필요합니다.
        self.nr = InitNornir(config_file=config_file)
        """
        # 모의 구현을 위한 가상 라우팅 테이블
        self.mock_routing_table = {
            "192.168.1.1": {  # gimpo-dmz-fw
                "100.130.23.45": "10.1.1.2"  # -> core-switch-1
            },
            "192.168.1.10": {  # core-switch-1
                "100.130.23.45": "10.0.0.6"  # -> yeouido-dev-l3
            },
            "192.168.1.2": {  # yeouido-dev-l3
                "100.130.23.45": "DIRECT"  # Directly connected
            },
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

        device_routes = self.mock_routing_table.get(management_ip)
        if not device_routes:
            return {}

        next_hop_ip = device_routes.get(destination_ip)
        if not next_hop_ip:
            return {}

        print(f"[MCP] 조회 결과: Next Hop은 {next_hop_ip} 입니다.")
        return {"next_hop_ip": next_hop_ip}


class MockMCP:
    """
    A mock Management and Control Plane (MCP) that simulates connecting to network devices
    and retrieving routing information. This avoids the need for live network equipment
    for development and testing.
    """

    def __init__(self):
        # This dictionary simulates the routing tables of our mock devices.
        # Key: management_ip, Value: { "destination_cidr": "next_hop_ip" }
        self.routing_tables = {
            # KIMPO_DMZ_L3_SWITCH (10.0.1.1)
            "10.0.1.1": {
                "100.130.23.0/24": "10.1.1.2",  # Route to Yeouido Dev -> via Core Router 1
                "default": "10.1.1.1",
            },
            # CORE_ROUTER_1 (10.0.0.1)
            "10.0.0.1": {
                "100.130.23.0/24": "10.1.1.6",  # Route to Yeouido Dev -> via Core Router 2
                "default": "0.0.0.0",  # Represents internet gateway
            },
            # CORE_ROUTER_2 (10.0.0.2)
            "10.0.0.2": {
                "100.130.23.0/24": "10.0.2.5",  # Route to Yeouido Dev -> Direct connection to Yeouido L3
                "default": "0.0.0.0",
            },
            # YEOUIDO_DEV_L3_SWITCH (10.0.2.1)
            "10.0.2.1": {
                # This device has a route for 100.150.50.x, but the next hop is a black hole.
                "100.150.50.0/24": "0.0.0.0",
                "default": "10.1.1.5",
            },
        }

    def get_next_hop(self, management_ip: str, destination_ip: str) -> str:
        """
        Simulates retrieving the next hop for a destination IP from a specific device.
        """
        print(f"MCP (Mock): Querying device {management_ip} for route to {destination_ip}")

        device_routes = self.routing_tables.get(management_ip)
        if not device_routes:
            return f"Error: No device found with management IP {management_ip} in mock setup."

        # A real implementation would use napalm's get_route_to.
        # Here, we just find the most specific matching route in our mock table.
        # For simplicity, we are doing an exact match on destination_ip for this mock.
        # A real longest-prefix match is more complex. Let's find a key that matches the dest.
        import ipaddress

        dest_addr = ipaddress.ip_address(destination_ip)

        best_match = None
        longest_prefix = -1

        for prefix_str, next_hop in device_routes.items():
            if prefix_str == "default":
                continue
            network = ipaddress.ip_network(prefix_str)
            if dest_addr in network:
                if network.prefixlen > longest_prefix:
                    longest_prefix = network.prefixlen
                    best_match = next_hop

        if best_match:
            return best_match

        return device_routes.get("default", "Error: No default route and no specific match found.")


# Singleton instance for easy access
mcp = MockMCP()
