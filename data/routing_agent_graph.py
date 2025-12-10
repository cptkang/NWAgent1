"""
LangGraph를 사용하여 라우팅 경로 추적 에이전트의 상태와 노드, 엣지를 정의합니다.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

# Allow running this module directly from the data/ folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.network_tools import (FindDeviceInfoTool, FindSubnetInfoTool,
                                   GetNextHopTool, ip_in_subnet)


class AgentState(TypedDict):
    """에이전트의 상태를 정의합니다."""
    user_prompt: str
    source_ip: str
    destination_ip: str
    # 추가된 정보
    source_location: Optional[str]
    destination_location: Optional[str]
    environment: Optional[str]
    source_device_info: Optional[Dict[str, Any]]
    source_device_name: Optional[str]
    destination_device_name: Optional[str]

    # 경로 추적 상태
    trace_path: List[Dict[str, Any]]
    current_hop_ip: str  # 다음으로 조사할 IP (시작은 source_ip)

    # 최종 결과 또는 오류
    final_answer: str
    error_message: str

class RoutingAgentGraph:
    """라우팅 경로 추적을 위한 LangGraph 워크플로우를 정의하고 생성합니다."""

    def __init__(self, llms: Dict[str, Any], tools: List[Any]):
        self.llms = llms
        # 각 노드에서 사용할 LLM 프로바이더를 정의합니다.
        # 'ollama' 또는 'fabrix' 등 main.py에서 초기화된 모델의 키를 사용합니다.
        self.node_llm_map = {
            "extract_initial_info": "ollama",
            "extract_initial_locate_info": "ollama",
            "summarize_path": "ollama",  # 예시: 요약은 ollama 사용
            "suggest_config": "ollama",  # 예시: 설정 제안은 ollama 사용
        }
        # 설정된 프로바이더를 찾을 수 없을 때 사용할 기본 프로바이더
        self.default_llm_provider = "ollama"

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
        graph.add_node("find_location_info_via_rag", self.find_location_info_via_rag)
        graph.add_node("extract_initial_locate_info", self.extract_initial_locate_info)
        graph.add_node("trace_route_step", self.trace_route_step)  # New main loop node
        graph.add_node("summarize_path", self.summarize_path)
        graph.add_node("suggest_config", self.suggest_config)
        graph.add_node("handle_error", self.handle_error)

        graph.set_entry_point("extract_initial_info")

        graph.add_conditional_edges(
            "extract_initial_info",
            self.decide_after_extraction,
            {"continue": "extract_initial_locate_info", "end": END},
        )

        graph.add_conditional_edges(
            "extract_initial_locate_info",
            self.decide_after_locate_info_extraction,
            {"find_via_rag": "find_location_info_via_rag", "continue_to_trace": "trace_route_step"},
        )

        graph.add_edge("find_location_info_via_rag", "trace_route_step")

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

    def _get_llm_for_node(self, node_name: str) -> Any:
        """
        노드 이름에 따라 구성된 LLM 클라이언트를 반환하는 헬퍼 함수.
        """
        provider = self.node_llm_map.get(node_name, self.default_llm_provider)
        llm = self.llms.get(provider)

        if not llm:
            print(f"경고: LLM 프로바이더 '{provider}'를 찾을 수 없어, 기본값 '{self.default_llm_provider}'로 대체합니다.")
            llm = self.llms.get(self.default_llm_provider)
            if not llm:
                raise ValueError(f"기본 LLM 프로바이더 '{self.default_llm_provider}'도 사용 불가능합니다.")
        
        print(f"--- (LLM 선택: 노드='{node_name}', 프로바이더='{provider}') ---")
        return llm

    def _extract_first_json(self, text: str) -> Optional[str]:
        """
        Extracts the first valid JSON string from a given text.
        Handles cases where JSON is wrapped in ```json ... ``` or appears directly.
        """
        # Try to find JSON block wrapped in ```json ... ```
        json_start_marker = "```json"
        json_end_marker = "```"
        
        start_code = text.find(json_start_marker)
        if start_code != -1:
            end_code = text.find(json_end_marker, start_code + len(json_start_marker))
            if end_code != -1:
                json_str = text[start_code + len(json_start_marker):end_code].strip()
                try:
                    json.loads(json_str) # Validate if it's actual JSON
                    return json_str
                except json.JSONDecodeError:
                    pass # Not a valid JSON inside the block, try other methods

        # Fallback: find the first '{' and try to find its matching '}'
        first_brace_idx = text.find('{')
        if first_brace_idx == -1:
            return None

        balance = 0
        for i in range(first_brace_idx, len(text)):
            if text[i] == '{':
                balance += 1
            elif text[i] == '}':
                balance -= 1
            if balance == 0 and text[i] == '}':
                json_str = text[first_brace_idx : i + 1]
                try:
                    json.loads(json_str) # Validate if it's actual JSON
                    return json_str
                except json.JSONDecodeError:
                    # This might happen if the first '}' is not the end of the top-level object
                    # or if there's garbage after it. Continue searching.
                    pass
        return None

    def extract_initial_info(self, state: AgentState) -> AgentState:
        """사용자 프롬프트에서 출발지/목적지 IP를 추출합니다."""
        print("--- 노드: 초기 IP 정보 추출 ---")
        llm = self._get_llm_for_node("extract_initial_info")
        prompt = f"""
        다음 사용자 질문에서 [출발지 IP]와 [목적지 IP]를 추출하여 JSON 형식으로 반환해줘.
        정보가 없으면 "정보 없음"으로 표시해줘. [출발지 IP]와 [목적지 IP]는 필수야.

        [예시1]
        질문: "김포센터 DMZ에 있는 100.120.50.40의 서버가 여의도센터 개발환경의 내부망에 있는 100.130.23.45와 통신하려고 한다. 라우팅 설정이 있는지 확인해줘."
        답변: {{"출발지 IP": "100.120.50.40", "목적지 IP": "100.130.23.45"}}
        [예시2]
        질문: "192.168.50.40이 100.150.50.40과 통신해야 된다. 라우팅 정보가 설정되어 있는지 확인해줘."
        답변: {{"출발지 IP": "192.168.50.40", "목적지 IP": "100.150.50.40"}}
        
        위의 "예시"에서 처럼 질문에 대한 답변이외에는 다른 답변을 하지 말아야 한다. 
        답변은 "예시"처럼 JSON형식으로 답변하라.  
        
        사용자 질문: "{state['user_prompt']}"        
        """
        response = llm.invoke([HumanMessage(content=prompt)])

        try:
            # LLM 응답에서 JSON 부분만 추출
            content = response.content
            json_str = self._extract_first_json(content)
            if not json_str:
                raise json.JSONDecodeError("No valid JSON found in LLM response", content, 0)
            info = json.loads(json_str)

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
                    "current_hop_ip": source_ip,  # 추적 시작점
                    "trace_path": [],
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"IP 정보 추출 실패: {e}")

        return {
            **state,
            "final_answer": "오류: 출발지 및 목적지 IP 주소를 식별할 수 없습니다. 명확한 IP 주소를 포함하여 다시 질문해주세요.",
        }

    def extract_initial_locate_info(self, state: AgentState) -> AgentState:
        """사용자 프롬프트에서 위치 및 환경 정보를 추출합니다."""
        print("--- 노드: 초기 위치 정보 추출 (LLM) ---")
        llm = self._get_llm_for_node("extract_initial_locate_info")
        prompt = f"""
        다음 사용자 질문에서 [출발지 위치정보], [목적지 위치정보], [환경정보]를 추출하여 JSON 형식으로 반환해줘.
        정보가 없으면 "정보 없음"으로 표시해줘.

        사용자 질문: "{state['user_prompt']}"

        [예시1]
        질문: "김포센터 DMZ에 있는 100.120.50.40의 서버가 여의도센터 개발환경의 내부망에 있는 100.130.23.45와 통신하려고 한다. 라우팅 설정이 있는지 확인해줘."
        답변: {{"출발지 위치정보": "김포센터 DMZ", "목적지 위치정보": "여의도센터 내부", "환경정보": "개발환경"}}
        [예시2]
        질문: "192.168.50.40이 100.150.50.40과 통신해야 된다. 라우팅 정보가 설정되어 있는지 확인해줘."
        답변: {{"출발지 위치정보": "정보 없음", "목적지 위치정보": "정보 없음", "환경정보": "정보 없음"}}
        
        [답변 기준]
        - 위의 "예시"에서 처럼 질문에 대한 답변이외에는 다른 답변을 하지 말아야 한다. 
        - 아래 기준에 맞게 위치정보와 환경정보에 대해 답변하라. 
          {{위치정보}} : 김포센터 DMZ, 김포센터 내부, 여의도센터 DMZ, 여의도센터 내부, 정보 없음
          {{환경정보}} : 운영환경, 개발환경, DR, 스테이징, 검증, 정보 없음
        
        [답변형식]
        - 답변은 "예시"처럼 JSON형식으로 답변하라.  
        - 답변에는 IP정보가 포함되면 안된다.
        
        
        사용자 질문: "{state['user_prompt']}"        
        """
        response = llm.invoke([HumanMessage(content=prompt)])

        try:
            # LLM 응답에서 JSON 부분만 추출
            json_str = self._extract_first_json(response.content)
            if not json_str:
                print(f"위치 정보 LLM 추출 실패: No valid JSON found in LLM response. Content: {response.content[:200]}...")
                return state # Return original state if no JSON is found
            info = json.loads(json_str)

            updated_state = {**state}
            source_loc = info.get("출발지 위치정보")
            dest_loc = info.get("목적지 위치정보")
            env = info.get("환경정보")

            if source_loc and source_loc != "정보 없음":
                updated_state["source_location"] = source_loc
            if dest_loc and dest_loc != "정보 없음":
                updated_state["destination_location"] = dest_loc
            if env and env != "정보 없음":
                updated_state["environment"] = env

            return updated_state

        except (json.JSONDecodeError, KeyError) as e:
            print(f"위치 정보 LLM 추출 실패 (RAG로 대체 시도): {e}. Content: {response.content[:200]}...")
            # 위치 정보 추출에 실패해도 RAG 노드로 분기하므로 계속 진행합니다.
            return state

    def decide_after_locate_info_extraction(self, state: AgentState) -> str:
        """LLM으로 위치 정보 추출 후 다음 단계를 결정합니다."""
        print("--- 엣지: 위치 정보 추출 후 분기 ---")
        source_loc = state.get("source_location")
        dest_loc = state.get("destination_location")

        # 만약 출발지 또는 목적지 위치 정보가 없다면 RAG를 통해 조회합니다.
        if (not source_loc or source_loc == "정보 없음") or \
           (not dest_loc or dest_loc == "정보 없음"):
            print("   - 위치 정보 불충분, RAG 검색으로 이동합니다.")
            return "find_via_rag"
        else:
            print("   - 위치 정보 충분, 경로 추적으로 이동합니다.")
            return "continue_to_trace"

    def find_location_info_via_rag(self, state: AgentState) -> AgentState:
        """RAG를 사용하여 IP의 위치, 환경, 장비 정보를 조회합니다."""
        print("--- 노드: 위치 및 장비 정보 조회 (RAG) ---")
        source_ip = state["source_ip"]
        destination_ip = state["destination_ip"]
        
        updated_state = {**state}

        # 출발지 정보 조회
        print(f"   [1] RAG로 {source_ip}의 서브넷 정보 조회...")
        src_subnet_info = self.find_subnet_info_tool._run(ip_address=source_ip)
        if not src_subnet_info.get("error"):
            # 위치 및 환경 정보 업데이트 (기존 정보가 없는 경우)
            if not updated_state.get("source_location") or updated_state.get("source_location") == "정보 없음":
                updated_state["source_location"] = src_subnet_info.get("location")
            if not updated_state.get("environment") or updated_state.get("environment") == "정보 없음":
                updated_state["environment"] = src_subnet_info.get("environment")
            
            # 장비 정보 및 serial_ip 조회 후 상태에 등록
            device_name = src_subnet_info.get("device_name")
            if device_name:
                print(f"   - 장비명({device_name})의 상세 정보 조회...")
            updated_state["source_device_name"] = device_name
                device_info = self.find_device_info_tool._run(device_name=device_name)
                if not device_info.get("error"):
                    updated_state["source_device_info"] = device_info
                    print(f"   - 출발지 장비 정보 저장 완료 (Serial IP: {device_info.get('serial_ip')})")

        # 목적지 정보 조회
        print(f"   [2] RAG로 {destination_ip}의 서브넷 정보 조회...")
        dest_subnet_info = self.find_subnet_info_tool._run(ip_address=destination_ip)
        if not dest_subnet_info.get("error"):
            # 목적지 위치 정보 업데이트 (기존 정보가 없는 경우)
            if not updated_state.get("destination_location") or updated_state.get("destination_location") == "정보 없음":
                updated_state["destination_location"] = dest_subnet_info.get("location")

            # 목적지 장비명 정보 업데이트
            dest_device_name = dest_subnet_info.get("device_name")
            if dest_device_name:
                updated_state["destination_device_name"] = dest_device_name
            print(f"   - 목적지 위치: {updated_state.get('destination_location')}, 장비명: {updated_state.get('destination_device_name')}")
        return updated_state

    def decide_after_extraction(self, state: AgentState) -> str:
        """IP 추출 후 다음 단계를 결정합니다."""
        print("--- 엣지: 추출 후 분기 ---")
        if state.get("final_answer"):
            return "end"
        return "continue"

    def trace_route_step(self, state: AgentState) -> AgentState:
        """경로의 한 단계를 추적하고 상태를 업데이트합니다."""
        is_first_hop = len(state["trace_path"]) == 0
        current_ip = state["source_ip"] if is_first_hop else state["current_hop_ip"]
        
        print(f"--- 노드: 경로 추적 단계 (현재 IP: {current_ip}) ---")

        # 1. 현재 IP에 대한 장비 정보 가져오기
        device_info = None
        if is_first_hop and state.get("source_device_info"):
            print("   - RAG로 사전 조회된 출발지 장비 정보 사용")
            device_info = state["source_device_info"]
        else:
            print(f"   - {current_ip}에 대한 장비 정보 RAG 조회...")
            subnet_info = self.find_subnet_info_tool._run(ip_address=current_ip)
            if subnet_info.get("error"):
                return {**state, "error_message": subnet_info["error"]}
            device_name = subnet_info.get("device_name")
            if not device_name:
                return {**state, "error_message": f"{current_ip}가 속한 장비를 찾을 수 없습니다."}
            device_info = self.find_device_info_tool._run(device_name=device_name)

        if not device_info or device_info.get("error"):
            return {**state, "error_message": device_info.get("error", f"{current_ip}에 대한 장비 정보를 가져올 수 없습니다.")}

        # 2. 장비 정보에서 Serial IP 추출
        device_name = device_info.get("device_name")
        serial_ip = device_info.get("serial_ip")
        if not serial_ip:
            return {**state, "error_message": f"장비 '{device_name}'의 Serial IP를 찾을 수 없습니다."}
        print(f"   - 장비명: {device_name}")
        print(f"   - Serial IP: {serial_ip} (MCP 통신에 사용)")

        # 3. 현재 경로 단계 기록 (serial_ip 포함)
        current_path_step = {
            "hop": len(state["trace_path"]) + 1,
            "device_name": device_name,
            "device_ip": current_ip,
            "management_ip": device_info.get("management_ip"),
            "serial_ip": serial_ip,
            "next_hop_ip": None,  # 아직 모름
            "error": None,
        }

        # 4. 목적지에 도달했는지 확인
        dest_subnet_info = self.find_subnet_info_tool._run(ip_address=state["destination_ip"])
        if dest_subnet_info.get("device_name") == device_name:
            print(f"   [성공] 목적지 장비 '{device_name}'에 도달했습니다.")
            current_path_step["next_hop_ip"] = "DIRECTLY_CONNECTED"
            return {**state, "trace_path": state["trace_path"] + [current_path_step]}

        # 5. Serial IP를 이용해 Next Hop 찾기
        print(f"   [3] {serial_ip}에서 목적지({state['destination_ip']})로의 Next Hop 조회...")
        next_hop_info = self.get_next_hop_tool._run(
            management_ip=serial_ip, destination_ip=state["destination_ip"]
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
        # 이 노드는 LLM을 사용하지 않고, 수집된 정보를 바탕으로 문자열을 조합합니다.
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
        llm = self._get_llm_for_node("suggest_config")
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
        response = llm.invoke([HumanMessage(content=prompt)])

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
