from langchain_core.tools import tool
from typing import Dict, Optional, Any
from mcp import mcp
from rag import network_info_db

@tool
def find_subnet_info_for_ip(ip_address: str) -> Optional[Dict[str, Any]]:
    """
    Finds the subnet information for a given IP address by searching the IP information database (RAG).
    This tool takes a single IP address and returns metadata about the subnet it belongs to, 
    including the device it's connected to.
    
    Args:
        ip_address: The 32-bit IP address to search for.
    
    Returns:
        A dictionary containing subnet details (subnet, affiliate, location, device, etc.) or None if not found.
    """
    print(f"TOOL: Searching for subnet info for IP: {ip_address}")
    return network_info_db.find_subnet_for_ip(ip_address)

@tool
def get_next_hop(management_ip: str, destination_ip: str) -> str:
    """
    Connects to a network device via its management IP and retrieves the next hop 
    in the path towards the given destination IP. This simulates querying a device's routing table.

    Args:
        management_ip: The management IP of the device to query.
        destination_ip: The ultimate destination IP of the traffic.

    Returns:
        The IP address of the next hop router, or a message indicating no route was found.
    """
    print(f"TOOL: Getting next hop from device {management_ip} for destination {destination_ip}")
    # In a real scenario, this would trigger a Nornir/Napalm task.
    # We use a mock MCP for this example.
    # from mcp import mcp
    return mcp.get_next_hop(management_ip, destination_ip)

# List of tools to be used by the agent
available_tools = [find_subnet_info_for_ip, get_next_hop]
