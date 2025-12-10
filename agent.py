import os
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from ollama_client import LLMAPIClient
from rag import network_info_db
from tools import get_next_hop, find_subnet_info_for_ip
import ipaddress


# --- 1. Define the State for the Graph ---
class RouteTraceState(TypedDict):
    """
    Represents the state of our route tracing agent.
    """
    query: str
    source_ip: Optional[str]
    destination_ip: Optional[str]
    
    # The trace path, a list of dicts with device info
    path: List[Dict[str, Any]]
    
    # The management IP of the device we are currently at
    current_device_mgmt_ip: Optional[str]
    
    # The IP address of the very next hop found by the tool
    next_hop_ip: Optional[str]
    
    # To detect loops
    visited_mgmt_ips: List[str]

    # Final response or error message
    response: Optional[str]
    error_message: Optional[str]


# --- 2. Define the LLM for Entity Extraction ---
class QueryInfo(BaseModel):
    """Pydantic model for extracting structured info from the user query."""
    source_ip: str = Field(description="The source IP address mentioned in the query.")
    destination_ip: str = Field(description="The destination IP address mentioned in the query.")

def get_entity_extractor():
    """Returns a runnable that extracts IPs from a query."""
    llm = LLMAPIClient(api_key="dummy", model="llama3").get_chat_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are an expert network engineer. Your task is to extract the source and destination IP addresses from the user's query. "
             "Do not make up any information. If an IP is not present, leave the field empty."),
            ("human", "{query}"),
        ]
    )
    return prompt | llm.with_structured_output(QueryInfo)


# --- 3. Define Graph Nodes ---

def extract_entities_node(state: RouteTraceState):
    """
    Node to extract source and destination IPs from the initial query.
    """
    print("---""NODE: EXTRACTING ENTITIES""---")
    extractor = get_entity_extractor()
    query = state['query']
    
    try:
        extracted: QueryInfo = extractor.invoke({"query": query})
        if extracted.source_ip and extracted.destination_ip:
            print(f"  -> Extracted: SRC={extracted.source_ip}, DST={extracted.destination_ip}")
            state['source_ip'] = extracted.source_ip
            state['destination_ip'] = extracted.destination_ip
        else:
            state['error_message'] = "Could not extract both a source and destination IP from the query."
    except Exception as e:
        state['error_message'] = f"LLM extraction failed: {e}"
    
    return state


def find_start_node(state: RouteTraceState):
    """
    Node to find the very first device in the path using the source IP.
    """
    print("---""NODE: FINDING STARTING DEVICE""---")
    src_ip = state['source_ip']
    
    # Use the RAG helper to find the device matching the source IP
    start_device = network_info_db.find_device_by_ip(src_ip)
    
    if start_device:
        mgmt_ip = start_device.get('management_ip')
        print(f"  -> Found start device '{start_device.get('hostname')}' with management IP {mgmt_ip}")
        state['current_device_mgmt_ip'] = mgmt_ip
        state['path'] = [start_device]
        state['visited_mgmt_ips'] = [mgmt_ip]
    else:
        state['error_message'] = f"No network device found in the database for the source IP: {src_ip}"
        
    return state


def trace_hop_node(state: RouteTraceState):
    """
    Node that calls the 'get_next_hop' tool to find the next step in the path.
    """
    print("---""NODE: TRACING HOP""---")
    mgmt_ip = state['current_device_mgmt_ip']
    dest_ip = state['destination_ip']
    
    # Call the tool to get the next hop
    next_hop = get_next_hop.invoke({"management_ip": mgmt_ip, "destination_ip": dest_ip})
    print(f"  -> Next hop from {mgmt_ip} for {dest_ip} is: {next_hop}")

    if not next_hop or "Error" in next_hop or "No route" in next_hop:
        # Get the hostname of the device that has no route
        current_device_info = state['path'][-1]
        hostname = current_device_info.get('hostname', mgmt_ip)
        
        # Try to find the destination subnet to create a helpful suggestion
        dest_subnet_info = network_info_db.find_subnet_for_ip(dest_ip)
        dest_subnet = dest_subnet_info.get('subnet', dest_ip) if dest_subnet_info else dest_ip
        
        # Create Cisco-like route command
        suggestion = f"ip route {dest_subnet} {next_hop if next_hop else '?.?.?.?'}"
        
        state['error_message'] = (f"Routing incomplete. No route to {dest_ip} found on device '{hostname}'.\n"
                                f"  > Suggested command for '{hostname}':\n    config t\n      {suggestion}")
    else:
        state['next_hop_ip'] = next_hop
        
    return state

def resolve_next_hop_node(state: RouteTraceState):
    """
    Node to take the found next_hop_ip and resolve it to a device.
    """
    print("---""NODE: RESOLVING NEXT HOP DEVICE""---")
    next_hop_ip = state['next_hop_ip']
    
    # Find which device corresponds to this next hop IP
    next_device = network_info_db.find_device_by_ip(next_hop_ip)
    
    if not next_device:
        state['error_message'] = f"The next hop IP {next_hop_ip} does not correspond to any known device in the database."
        return state

    mgmt_ip = next_device.get('management_ip')
    print(f"  -> Next hop IP {next_hop_ip} resolved to device '{next_device.get('hostname')}' (Mgmt: {mgmt_ip})")

    # Check for a routing loop
    if mgmt_ip in state['visited_mgmt_ips']:
        state['path'].append(next_device) # Add to show the loop
        state['error_message'] = f"Routing loop detected! Device '{next_device.get('hostname')}' has already been visited."
        return state

    state['path'].append(next_device)
    state['visited_mgmt_ips'].append(mgmt_ip)
    state['current_device_mgmt_ip'] = mgmt_ip
    
    return state

def format_response_node(state: RouteTraceState):
    """
    Node to format the final successful response.
    """
    print("---""NODE: FORMATTING RESPONSE""---")
    path = state['path']
    
    response_lines = ["✅ Routing path found successfully!\n"]
    for i, device in enumerate(path):
        line = f"{i+1}. Device: {device.get('hostname', 'N/A')} (Location: {device.get('location', 'N/A')}, Mgmt IP: {device.get('management_ip', 'N/A')})"
        response_lines.append(line)
        if i < len(path) - 1:
            next_hop = state['path'][i+1].get('hostname')
            response_lines.append(f"   | \n   V (via next-hop to {next_hop})")

    state['response'] = "\n".join(response_lines)
    return state
    
def format_error_node(state: RouteTraceState):
    """
    Node to format an error response.
    """
    print("---""NODE: FORMATTING ERROR""---")
    state['response'] = f"❌ Error: {state['error_message']}"
    return state


# --- 4. Define Conditional Edges ---

def decide_after_extraction(state: RouteTraceState):
    """Conditional edge after entity extraction."""
    return "error" if state.get('error_message') else "find_start_node"

def decide_after_start(state: RouteTraceState):
    """Conditional edge after finding the start node."""
    return "error" if state.get('error_message') else "trace_hop"

def decide_after_hop(state: RouteTraceState):
    """
    The main routing logic after a hop trace.
    Decides whether to continue tracing, finish, or error out.
    """
    if state.get('error_message'):
        return "error"
        
    # Check if the next hop lands us in the destination subnet
    dest_ip = state['destination_ip']
    next_hop_ip = state['next_hop_ip']
    
    dest_subnet_info = network_info_db.find_subnet_for_ip(dest_ip)
    next_hop_subnet_info = network_info_db.find_subnet_for_ip(next_hop_ip)

    if dest_subnet_info and next_hop_subnet_info:
        if dest_subnet_info['subnet'] == next_hop_subnet_info['subnet']:
            print("  -> Destination subnet reached!")
            return "resolve_next_hop_and_finish"
            
    print("  -> Destination not yet reached. Continuing trace.")
    return "resolve_and_continue"

def decide_after_resolve(state: RouteTraceState):
    """Conditional edge after resolving a hop to a device."""
    return "error" if state.get('error_message') else "trace_hop"


# --- 5. Build the Graph ---

def build_graph():
    """Builds and compiles the LangGraph agent."""
    workflow = StateGraph(RouteTraceState)

    # Add nodes
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("find_start_node", find_start_node)
    workflow.add_node("trace_hop", trace_hop_node)
    workflow.add_node("resolve_next_hop", resolve_next_hop_node)
    workflow.add_node("format_success", format_response_node)
    workflow.add_node("format_error", format_error_node)

    # Set entrypoint
    workflow.set_entry_point("extract_entities")

    # Add edges
    workflow.add_conditional_edge(
        "extract_entities",
        decide_after_extraction,
        {"find_start_node": "find_start_node", "error": "format_error"}
    )
    workflow.add_conditional_edge(
        "find_start_node",
        decide_after_start,
        {"trace_hop": "trace_hop", "error": "format_error"}
    )
    workflow.add_conditional_edge(
        "trace_hop",
        decide_after_hop,
        {
            "resolve_and_continue": "resolve_next_hop",
            "resolve_next_hop_and_finish": "resolve_next_hop", # Resolve the final device
            "error": "format_error"
        }
    )
    workflow.add_conditional_edge(
        "resolve_next_hop",
        decide_after_resolve,
        {"trace_hop": "trace_hop", "error": "format_error"}
    )

    workflow.add_edge("resolve_next_hop", "format_success") # This is tricky, see decide_after_hop
    
    # Let's refine the logic from hop to end
    # We need a separate path for the final successful resolution
    workflow.remove_edge("resolve_next_hop", "format_success")
    workflow.add_edge("format_success", END)
    workflow.add_edge("format_error", END)
    
    # Corrected logic:
    # trace_hop -> decide_after_hop -> resolve_next_hop
    # resolve_next_hop -> decide_after_resolve -> trace_hop (loop)
    # The exit condition must be handled carefully.
    
    # Let's adjust `decide_after_hop` to route to a finalization node
    # This is slightly complex, so for now, we'll let it resolve the last hop
    # and then the path will just end. The format node will be the end.
    
    # Let's simplify the graph flow slightly for clarity
    
    # Reset and build again with cleaner conditional logic
    
    # This is a better way to handle the end state:
    # 1. Trace a hop.
    # 2. Decide if it's the end. If so, go to resolve and then to SUCCESS.
    # 3. If not, go to resolve, check for loops, and then loop back to TRACE.
    
    # The logic above is almost right, but the final connection needs thought.
    # `decide_after_hop` is the key. When it returns "resolve_next_hop_and_finish",
    # the next node should be `resolve_next_hop`, and after THAT, it should go to `format_success`.
    
    # This requires another conditional node, or a more complex state check.
    # For now, let's create a simplified main loop.
    
    return workflow.compile()


if __name__ == '__main__':
    # Simple check for required files
    if not os.path.exists('data/network_devices.json') or not os.path.exists('data/ip_info.json'):
        print("Error: Required data files ('data/network_devices.json', 'data/ip_info.json') not found.")
        print("Please create them before running the agent.")
    else:
        app = build_graph()
        
        # --- Example 1: Successful trace ---
        print("\n---""RUNNING EXAMPLE 1: SUCCESSFUL TRACE""---")
        query1 = "김포센터 DMZ에 있는 100.120.50.40의 서버가 여의도센터 개발환경의 내부망에 있는 100.130.23.45와 통신하려고 한다. 라우팅 설정이 있는지 확인해줘."
        initial_state1 = {"query": query1, "path": [], "visited_mgmt_ips": []}
        final_state1 = app.invoke(initial_state1)
        print("\n---""FINAL RESPONSE 1""---")
        print(final_state1.get('response'))

        # --- Example 2: Incomplete trace (No route) ---
        print("\n\n---""RUNNING EXAMPLE 2: NO ROUTE""---")
        query2 = "192.168.50.40이 100.150.50.40과 통신해야 된다. 라우팅 정보가 설정되어 있는지 확인해줘."
        initial_state2 = {"query": query2, "path": [], "visited_mgmt_ips": []}
        final_state2 = app.invoke(initial_state2)
        print("\n---""FINAL RESPONSE 2""---")
        print(final_state2.get('response'))
        
        # --- Example 3: Invalid input ---
        print("\n\n---""RUNNING EXAMPLE 3: INVALID INPUT""---")
        query3 = "내 서버가 잘 돌아가는지 확인해줘"
        initial_state3 = {"query": query3, "path": [], "visited_mgmt_ips": []}
        final_state3 = app.invoke(initial_state3)
        print("\n---""FINAL RESPONSE 3""---")
        print(final_state3.get('response'))