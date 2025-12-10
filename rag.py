import json
import ipaddress
from typing import Dict, List, Optional, Any

class NetworkInfo:
    """
    Manages loading and querying network information from JSON files.
    This class acts as the RAG component for structured data.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(NetworkInfo, cls).__new__(cls)
        return cls._instance

    def __init__(self, devices_path: str = 'data/network_devices.json', ip_info_path: str = 'data/ip_info.json'):
        # The __init__ will be called every time, but the state is shared via _instance
        if not hasattr(self, 'initialized'):
            self.devices_path = devices_path
            self.ip_info_path = ip_info_path
            self.devices: List[Dict[str, Any]] = self._load_json(self.devices_path)
            self.ip_info: List[Dict[str, Any]] = self._load_json(self.ip_info_path)
            self.initialized = True

    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Data file not found at {file_path}. Returning empty list.")
            return []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Returning empty list.")
            return []

    def find_device_by_hostname(self, hostname: str) -> Optional[Dict[str, Any]]:
        """Finds a device's information by its hostname."""
        for device in self.devices:
            if device.get('hostname') == hostname:
                return device
        return None

    def find_subnet_for_ip(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """
        Finds the subnet information for a given IP address by checking against
        the list of CIDR subnets.
        """
        try:
            target_ip = ipaddress.ip_address(ip_address)
        except ValueError:
            return None  # Invalid IP address format

        for subnet_info in self.ip_info:
            subnet_cidr = subnet_info.get('subnet')
            if not subnet_cidr:
                continue
            
            try:
                network = ipaddress.ip_network(subnet_cidr)
                if target_ip in network:
                    return subnet_info
            except ValueError:
                # Invalid CIDR format in the data file
                continue
        
        return None

    def find_device_by_ip(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """
        Finds the full device information (including management IP) for a given IP address.
        """
        subnet_info = self.find_subnet_for_ip(ip_address)
        if subnet_info:
            device_hostname = subnet_info.get('device')
            if device_hostname:
                return self.find_device_by_hostname(device_hostname)
        return None

# Singleton instance for easy access across the application
network_info_db = NetworkInfo()