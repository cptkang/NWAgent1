"""
RoutingAgentGraph의 단위 테스트 코드
LLM과 Tool들을 모의(Mock) 객체로 대체하여 그래프의 로직을 검증합니다.
"""
import json
import unittest
from unittest.mock import MagicMock

# 테스트 대상 클래스 및 상태
from routing_agent_graph import AgentState, RoutingAgentGraph
# 모의 객체로 대체할 의존성
from mcp import RoutingMCP
from ollama_client import CustomOllamaChat
from tools.network_tools import (FindDeviceInfoTool, FindSubnetInfoTool,
                                 GetNextHopTool)


class TestRoutingAgentGraph(unittest.TestCase):
    """RoutingAgentGraph의 테스트 스위트"""

    def setUp(self):
        """각 테스트 실행 전에 모의 객체들을 설정합니다."""
        # 1. LLM 모의 객체 생성
        self.mock_llm = MagicMock(spec=CustomOllamaChat)

        # 2. Tool 모의 객체 생성
        # 실제 Tool 객체를 생성하되, 내부 의존성(retriever, mcp)은 모의 객체로 처리합니다.
        mock_retriever = MagicMock()
        mock_mcp = MagicMock(spec=RoutingMCP)

        self.find_subnet_tool = FindSubnetInfoTool(ip_subnet_retriever=mock_retriever)
        self.find_device_tool = FindDeviceInfoTool(device_info_retriever=mock_retriever)
        self.get_next_hop_tool = GetNextHopTool(mcp=mock_mcp)

        # 각 테스트에서 `_run` 메서드의 반환 값을 설정할 수 있도록 모의 객체로 교체합니다.
        self.find_subnet_tool._run = MagicMock()
        self.find_device_tool._run = MagicMock()
        self.get_next_hop_tool._run = MagicMock()

        tools = [self.find_subnet_tool, self.find_device_tool, self.get_next_hop_tool]

        # 3. 테스트 대상인 RoutingAgentGraph 인스턴스 생성
        self.agent_graph = RoutingAgentGraph(self.mock_llm, tools).graph

    def _configure_llm_for_extraction(self, src_ip, dest_ip):
        """정보 추출 노드에서 사용할 LLM 모의 객체의 반환 값을 설정하는 헬퍼 함수"""
        response_content = json.dumps({
            "출발지 IP": src_ip, "목적지 IP": dest_ip,
            "출발지 위치정보": "정보 없음", "목적지 위치정보": "정보 없음", "환경정보": "정보 없음"
        })
        mock_response = MagicMock()
        mock_response.content = f"```json\n{response_content}\n```"
        self.mock_llm.invoke.return_value = mock_response

    def test_successful_trace(self):
        """성공적인 경로 추적 시나리오를 테스트합니다."""
        print("\n--- Running test_successful_trace ---")
        # 1. 모의 객체 설정
        self._configure_llm_for_extraction("100.120.50.40", "100.130.23.45")

        # 각 Hop에서 호출될 Tool들의 반환 값을 순서대로 설정
        self.find_subnet_tool._run.side_effect = [
            {"device_name": "gimpo-dmz-fw"},  # Hop 1 (현재 IP)
            {"device_name": "yeouido-dev-l3"}, # Hop 1 (목적지 IP 확인)
            {"device_name": "core-switch-1"}, # Hop 2 (현재 IP)
            {"device_name": "yeouido-dev-l3"}, # Hop 2 (목적지 IP 확인)
            {"device_name": "yeouido-dev-l3"}, # Hop 3 (현재 IP)
            {"device_name": "yeouido-dev-l3"}, # Hop 3 (목적지 IP 확인 -> 일치)
        ]
        self.find_device_tool._run.side_effect = [
            {"management_ip": "192.168.1.1"},  # gimpo-dmz-fw
            {"management_ip": "192.168.1.10"}, # core-switch-1
            {"management_ip": "192.168.1.2"},   # yeouido-dev-l3
        ]
        self.get_next_hop_tool._run.side_effect = [
            {"next_hop_ip": "10.1.1.2"},  # from gimpo-dmz-fw
            {"next_hop_ip": "10.0.0.6"},  # from core-switch-1
        ]

        # 2. 초기 상태 정의
        initial_state: AgentState = {
            "user_prompt": "100.120.50.40에서 100.130.23.45로 가는 경로 확인해줘.",
            "source_ip": "", "destination_ip": "", "source_location": None,
            "destination_location": None, "environment": None, "trace_path": [],
            "current_hop_ip": "", "final_answer": "", "error_message": ""
        }

        # 3. 그래프 실행
        final_state = self.agent_graph.invoke(initial_state)

        # 4. 결과 검증
        self.assertIn("라우팅 경로 추적 완료", final_state["final_answer"])
        self.assertIn("Hop 1: gimpo-dmz-fw", final_state["final_answer"])
        self.assertIn("Next Hop: 10.1.1.2", final_state["final_answer"])
        self.assertIn("Hop 2: core-switch-1", final_state["final_answer"])
        self.assertIn("Next Hop: 10.0.0.6", final_state["final_answer"])
        self.assertIn("Hop 3: yeouido-dev-l3", final_state["final_answer"])
        self.assertIn("성공적으로 도달했습니다", final_state["final_answer"])
        self.assertEqual(final_state.get("error_message"), "")

    def test_no_route_found(self):
        """경로가 없어 추적이 실패하는 시나리오를 테스트합니다."""
        print("\n--- Running test_no_route_found ---")
        # 1. 모의 객체 설정
        self._configure_llm_for_extraction("192.168.50.40", "100.150.50.40")
        # 설정 제안을 위한 LLM 호출 모의
        self.mock_llm.invoke.side_effect = [
            self.mock_llm.invoke.return_value,
            MagicMock(content="ip route 100.150.50.0 255.255.255.0 ?.?.?.?")
        ]

        self.find_subnet_tool._run.side_effect = [
            {"device_name": "gimpo-internal-fw"}, # Hop 1 (현재 IP)
            {"device_name": "yeouido-dmz-fw"},    # Hop 1 (목적지 IP 확인)
            {"subnet": "100.150.50.0/24"},       # 설정 제안 노드에서 사용
        ]
        self.find_device_tool._run.return_value = {"management_ip": "192.168.1.3"}
        self.get_next_hop_tool._run.return_value = {"error": "No route to host"}

        # 2. 초기 상태 정의
        initial_state: AgentState = {
            "user_prompt": "192.168.50.40에서 100.150.50.40으로 가는 경로 확인해줘.",
            "source_ip": "", "destination_ip": "", "source_location": None,
            "destination_location": None, "environment": None, "trace_path": [],
            "current_hop_ip": "", "final_answer": "", "error_message": ""
        }

        # 3. 그래프 실행
        final_state = self.agent_graph.invoke(initial_state)

        # 4. 결과 검증
        self.assertIn("라우팅 경로 추적 실패", final_state["final_answer"])
        self.assertIn("'gimpo-internal-fw' 장비에서 목적지", final_state["final_answer"])
        self.assertIn("필요한 설정 예시", final_state["final_answer"])
        self.assertIn("ip route 100.150.50.0", final_state["final_answer"])

    def test_invalid_input_no_ips(self):
        """초기 질문에 IP 정보가 없는 시나리오를 테스트합니다."""
        print("\n--- Running test_invalid_input_no_ips ---")
        # 1. 모의 객체 설정
        self._configure_llm_for_extraction("정보 없음", "정보 없음")

        # 2. 초기 상태 정의
        initial_state: AgentState = {
            "user_prompt": "내 서버가 안돼",
            "source_ip": "", "destination_ip": "", "source_location": None,
            "destination_location": None, "environment": None, "trace_path": [],
            "current_hop_ip": "", "final_answer": "", "error_message": ""
        }

        # 3. 그래프 실행
        final_state = self.agent_graph.invoke(initial_state)

        # 4. 결과 검증
        self.assertIn("출발지 및 목적지 IP 주소를 식별할 수 없습니다", final_state["final_answer"])

    def test_loop_detection(self):
        """라우팅 루프가 감지되는 시나리오를 테스트합니다."""
        print("\n--- Running test_loop_detection ---")
        # 1. 모의 객체 설정
        self._configure_llm_for_extraction("1.1.1.1", "2.2.2.2")

        self.find_subnet_tool._run.side_effect = [
            {"device_name": "device-A"}, {"device_name": "device-C"}, # Hop 1
            {"device_name": "device-B"}, {"device_name": "device-C"}, # Hop 2
        ]
        self.find_device_tool._run.side_effect = [
            {"management_ip": "10.0.0.1"}, # device-A
            {"management_ip": "10.0.0.2"}, # device-B
        ]
        self.get_next_hop_tool._run.side_effect = [
            {"next_hop_ip": "3.3.3.3"}, # from device-A to device-B's IP
            {"next_hop_ip": "1.1.1.1"}, # from device-B back to device-A's IP
        ]

        # 2. 초기 상태 정의
        initial_state: AgentState = {
            "user_prompt": "1.1.1.1에서 2.2.2.2로",
            "source_ip": "", "destination_ip": "", "source_location": None,
            "destination_location": None, "environment": None, "trace_path": [],
            "current_hop_ip": "", "final_answer": "", "error_message": ""
        }

        # 3. 그래프 실행
        final_state = self.agent_graph.invoke(initial_state)

        # 4. 결과 검증
        self.assertIn("라우팅 루프 감지됨", final_state["final_answer"])
        self.assertIn("3.3.3.3 주소를 다시 방문했습니다", final_state["final_answer"])


if __name__ == '__main__':
    unittest.main()
