"""
LangGraph를 사용하여 라우팅 경로 추적 에이전트의 상태와 노드, 엣지를 정의합니다.
"""
import json
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from tools.network_tools import (
    FindDeviceInfoTool,
    FindSubnetInfoTool,
    GetNextHopTool,
    ip_in_subnet,
)


class AgentState(TypedDict):
    """에이전트의 상태를 정의합니다."""
    user_prompt: str
    source_ip: str
    destination_ip: str
    # 추가된 정보
    source_location: Optional[str]
    destination_location: Optional[str]
    environment: Optional[str]

    # 경로 추적 상태
    trace_path: List[Dict[str, Any]]
    current_hop_ip: str  # 다음으로 조사할 IP (시작은 source_ip)

    # 최종 결과 또는 오류
    final_answer: str
    error_message: str

class RoutingAgentGraph:
    """라우팅 경로 추적을 위한 LangGraph 워크플로우를 정의하고 생성합니다."""

    def __init__(self, llm, tools):
        self.llm = llm  # For extraction and summarization
        # Store tools for direct access
        self.find_subnet_info_tool = next(
            (t for t in tools if isinstance(t, FindSubnetInfoTool)), None
        )
        self.find_device_info_tool = next(
            (t for t in tools if isinstance(t, FindDeviceInfoTool)), None
        )
        self.get_next_hop_tool = next(
            (t for t in tools if isinstance(t, GetNextHopTool)), None
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """상태 그래프를 구성합니다."""
        graph = StateGraph(AgentState)

        graph.add_node("extract_initial_info", self.extract_initial_info)
        graph.add_node("trace_route_step", self.trace_route_step)  # New main loop node
        graph.add_node("summarize_path", self.summarize_path)
        graph.add_node("suggest_config", self.suggest_config)
        graph.add_node("handle_error", self.handle_error)

        graph.set_entry_point("extract_initial_info")

        graph.add_conditional_edges(
            "extract_initial_info",
            self.decide_after_extraction,
            {"continue": "trace_route_step", "end": END},
        )

        graph.add_conditional_edges(
            "trace_route_step",
            self.decide_after_trace,
            {
                "continue": "trace_route_step",  # Loop back
                "summarize": "summarize_path",
                "suggest_config": "suggest_config",
                "error": "handle_error",
            },
        )

        graph.add_edge("summarize_path", END)
        graph.add_edge("suggest_config", END)
        graph.add_edge("handle_error", END)

        return graph.compile()

    def extract_initial_info(self, state: AgentState) -> AgentState:
        """사용자 프롬프트에서 출발지/목적지 IP를 추출합니다."""
        print("--- 노드: 초기 정보 추출 ---")
        prompt = f"""
        다음 사용자 질문에서 [출발지 IP], [목적지 IP], [출발지 위치정보], [목적지 위치정보], [환경정보]를 추출하여 JSON 형식으로 반환해줘.
        정보가 없으면 "정보 없음"으로 표시해줘. 특히 [출발지 IP]와 [목적지 IP]는 필수야.

        사용자 질문: "{state['user_prompt']}"

        예시:
        질문: "김포센터 DMZ에 있는 100.120.50.40의 서버가 여의도센터 개발환경의 내부망에 있는 100.130.23.45와 통신하려고 한다. 라우팅 설정이 있는지 확인해줘."
        답변: {{'출발지 IP': '100.120.50.40', '목적지 IP': '100.130.23.45', '출발지 위치정보': '김포센터 DMZ', '목적지 위치정보': '여의도센터 내부', '환경정보': '개발환경'}}
        
        질문: "192.168.50.40이 100.150.50.40과 통신해야 된다. 라우팅 정보가 설정되어 있는지 확인해줘."
        답변: {{'출발지 IP': '192.168.50.40', '목적지 IP': '100.150.50.40', '출발지 위치정보': '정보 없음', '목적지 위치정보': '정보 없음', '환경정보': '정보 없음'}}
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            # LLM 응답에서 JSON 부분만 추출
            content = response.content
            json_part = content[content.find("{") : content.rfind("}") + 1]
            info = json.loads(json_part)

            source_ip = info.get("출발지 IP")
            destination_ip = info.get("목적지 IP")

            if (
                source_ip
                and source_ip != "정보 없음"
                and destination_ip
                and destination_ip != "정보 없음"
            ):
                return {
                    **state,
                    "source_ip": source_ip,
                    "destination_ip": destination_ip,
                    "source_location": info.get("출발지 위치정보"),
                    "destination_location": info.get("목적지 위치정보"),
                    "environment": info.get("환경정보"),
                    "current_hop_ip": source_ip,  # 추적 시작점
                    "trace_path": [],
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"IP 정보 추출 실패: {e}")

        return {
            **state,
            "final_answer": "오류: 출발지 및 목적지 IP 주소를 식별할 수 없습니다. 명확한 IP 주소를 포함하여 다시 질문해주세요.",
        }

    def decide_after_extraction(self, state: AgentState) -> str:
        """IP 추출 후 다음 단계를 결정합니다."""
        print("--- 엣지: 추출 후 분기 ---")
        if state.get("final_answer"):
            return "end"
        return "continue"

    def trace_route_step(self, state: AgentState) -> AgentState:
        """경로의 한 단계를 추적하고 상태를 업데이트합니다."""
        current_ip = state["current_hop_ip"]
        print(f"--- 노드: 경로 추적 단계 (현재 IP: {current_ip}) ---")

        # 1. 현재 IP가 속한 장비 정보 찾기
        print(f"   [1] {current_ip}의 서브넷 정보 조회...")
        subnet_info = self.find_subnet_info_tool._run(ip_address=current_ip)
        if subnet_info.get("error"):
            return {**state, "error_message": subnet_info["error"]}

        device_name = subnet_info.get("device_name")
        if not device_name:
            return {**state, "error_message": f"{current_ip}가 속한 장비를 찾을 수 없습니다."}
        print(f"   - 장비명: {device_name}")

        # 2. 장비명으로 Management IP 찾기
        print(f"   [2] {device_name}의 상세 정보 조회...")
        device_info = self.find_device_info_tool._run(device_name=device_name)
        if device_info.get("error"):
            return {**state, "error_message": device_info["error"]}

        management_ip = device_info.get("management_ip")
        if not management_ip:
            return {**state, "error_message": f"{device_name}의 Management IP를 찾을 수 없습니다."}
        print(f"   - Management IP: {management_ip}")

        # 3. 현재 경로 단계 기록
        current_path_step = {
            "hop": len(state["trace_path"]) + 1,
            "device_name": device_name,
            "device_ip": current_ip,  # 이 장비에 도달하게 한 IP
            "management_ip": management_ip,
            "next_hop_ip": None,  # 아직 모름
            "error": None,
        }

        # 4. 목적지에 도달했는지 확인
        # 현재 장비가 목적지 IP를 포함하는 서브넷에 연결되어 있는지 확인
        dest_subnet_info = self.find_subnet_info_tool._run(
            ip_address=state["destination_ip"]
        )
        if dest_subnet_info.get("device_name") == device_name:
            print(f"   [성공] 목적지 장비 '{device_name}'에 도달했습니다.")
            current_path_step["next_hop_ip"] = "DIRECTLY_CONNECTED"
            return {**state, "trace_path": state["trace_path"] + [current_path_step]}

        # 5. Management IP를 이용해 Next Hop 찾기
        print(
            f"   [3] {management_ip}에서 목적지({state['destination_ip']})로의 Next Hop 조회..."
        )
        next_hop_info = self.get_next_hop_tool._run(
            management_ip=management_ip, destination_ip=state["destination_ip"]
        )

        if next_hop_info.get("error"):
            print(f"   [실패] 경로 없음: {next_hop_info['error']}")
            current_path_step["error"] = next_hop_info["error"]
            return {**state, "trace_path": state["trace_path"] + [current_path_step]}

        next_hop_ip = next_hop_info.get("next_hop_ip")
        print(f"   - Next Hop IP: {next_hop_ip}")
        current_path_step["next_hop_ip"] = next_hop_ip

        # 다음 스텝을 위해 current_hop_ip 업데이트
        return {
            **state,
            "trace_path": state["trace_path"] + [current_path_step],
            "current_hop_ip": next_hop_ip,
        }

    def decide_after_trace(self, state: AgentState) -> str:
        """경로 추적 한 단계 후 다음 행동을 결정합니다."""
        print("--- 엣지: 추적 후 분기 ---")
        if state.get("error_message"):
            return "error"

        last_step = state["trace_path"][-1]

        # 경로 추적 중 오류 발생 (경로 없음)
        if last_step.get("error"):
            return "suggest_config"

        next_hop_ip = last_step.get("next_hop_ip")

        # 목적지 도달
        if next_hop_ip == "DIRECTLY_CONNECTED":
            return "summarize"

        # 무한 루프 방지 (이미 방문한 IP를 다시 방문하는 경우)
        visited_ips = [s["device_ip"] for s in state["trace_path"]]
        if next_hop_ip in visited_ips:
            state[
                "error_message"
            ] = f"라우팅 루프 감지됨: {next_hop_ip} 주소를 다시 방문했습니다."
            return "error"

        # 최대 홉 수 초과
        if len(state["trace_path"]) > 10:
            state["error_message"] = "최대 홉(10)을 초과하여 추적을 중단합니다."
            return "error"

        # 추적 계속
        return "continue"

    def summarize_path(self, state: AgentState) -> AgentState:
        """성공적으로 찾은 경로를 요약하여 보고합니다."""
        print("--- 노드: 경로 요약 ---")
        path_str_list = []
        for s in state["trace_path"]:
            hop_str = f"Hop {s['hop']}: {s['device_name']} (도달 IP: {s['device_ip']})"
            if s["next_hop_ip"] and s["next_hop_ip"] != "DIRECTLY_CONNECTED":
                hop_str += f" -> Next Hop: {s['next_hop_ip']}"
            path_str_list.append(hop_str)

        path_str = "\n".join(path_str_list)
        summary = (
            f"라우팅 경로 추적 완료:\n\n"
            f"출발지: {state['source_ip']}\n"
            f"목적지: {state['destination_ip']}\n\n"
            f"경로:\n{path_str}\n\n"
            f"최종 목적지({state['destination_ip']})에 성공적으로 도달했습니다."
        )
        return {**state, "final_answer": summary}

    def suggest_config(self, state: AgentState) -> AgentState:
        """경로가 없을 때 설정 명령어를 제안합니다."""
        print("--- 노드: 설정 제안 ---")
        last_step = state["trace_path"][-1]
        missing_route_device = last_step["device_name"]
        management_ip = last_step["management_ip"]

        # 목적지 IP의 서브넷 정보 조회
        dest_subnet_info = self.find_subnet_info_tool._run(
            ip_address=state["destination_ip"]
        )
        dest_subnet = dest_subnet_info.get("subnet", state["destination_ip"] + "/32")

        prompt = f"""
        네트워크 장비 '{missing_route_device}'(관리 IP: {management_ip})에서 목적지 대역 '{dest_subnet}'(목적지 IP: {state['destination_ip']})로 가는 라우팅 경로가 없습니다.
        이 장비에 추가해야 할 라우팅 설정 명령어를 Cisco IOS 형식으로 생성해주세요. 
        Next-hop IP는 '?.?.?.?'로 표시해주세요.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])

        path_str = "\n".join(
            [
                f"Hop {s['hop']}: {s['device_name']} (도달 IP: {s['device_ip']}) -> Next Hop: {s.get('next_hop_ip', 'N/A')}"
                for s in state["trace_path"][:-1]
            ]
        )

        suggestion = (
            f"라우팅 경로 추적 실패: '{missing_route_device}' 장비에서 목적지({state['destination_ip']})로 가는 경로를 찾을 수 없습니다.\n\n"
            f"현재까지의 경로:\n{path_str}\n\n"
            f"'{missing_route_device}' 장비에 필요한 설정 예시:\n"
            f"```\n"
            f"{response.content}\n"
            f"```"
        )
        return {**state, "final_answer": suggestion}

    def handle_error(self, state: AgentState) -> AgentState:
        """오류 상태를 처리합니다."""
        print("--- 노드: 오류 처리 ---")
        error_msg = state.get("error_message", "알 수 없는 오류가 발생했습니다.")
        return {**state, "final_answer": f"오류가 발생하여 경로 추적을 중단합니다: {error_msg}"}
