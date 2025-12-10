import unittest
import os
import json
from rag import NetworkInfo

class TestNetworkInfo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory and dummy data files for testing."""
        cls.test_dir = "temp_test_data"
        os.makedirs(cls.test_dir, exist_ok=True)

        cls.devices_path = os.path.join(cls.test_dir, "test_devices.json")
        cls.ip_info_path = os.path.join(cls.test_dir, "test_ip_info.json")

        devices_data = [
            {"hostname": "ROUTER_A", "management_ip": "10.0.0.1", "location": "DC1"},
            {"hostname": "ROUTER_B", "management_ip": "10.0.0.2", "location": "DC2"}
        ]
        ip_info_data = [
            {"subnet": "192.168.1.0/24", "device": "ROUTER_A", "location": "DC1_INTERNAL"},
            {"subnet": "172.16.0.0/16", "device": "ROUTER_B", "location": "DC2_DMZ"}
        ]

        with open(cls.devices_path, 'w') as f:
            json.dump(devices_data, f)
        with open(cls.ip_info_path, 'w') as f:
            json.dump(ip_info_data, f)

        # Instantiate the NetworkInfo with test data paths
        cls.network_info = NetworkInfo(devices_path=cls.devices_path, ip_info_path=cls.ip_info_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary files and directory."""
        os.remove(cls.devices_path)
        os.remove(cls.ip_info_path)
        os.rmdir(cls.test_dir)

    def test_singleton(self):
        """Test that the NetworkInfo class is a singleton."""
        instance1 = NetworkInfo(devices_path=self.devices_path, ip_info_path=self.ip_info_path)
        instance2 = NetworkInfo()
        self.assertIs(instance1, instance2)
        # Reset to original data paths for other tests
        instance1.initialized = False
        instance1.__init__(devices_path='data/network_devices.json', ip_info_path='data/ip_info.json')


    def test_find_device_by_hostname(self):
        """Test finding a device by its hostname."""
        device = self.network_info.find_device_by_hostname("ROUTER_A")
        self.assertIsNotNone(device)
        self.assertEqual(device["management_ip"], "10.0.0.1")

        device_none = self.network_info.find_device_by_hostname("NON_EXISTENT")
        self.assertIsNone(device_none)

    def test_find_subnet_for_ip(self):
        """Test finding a subnet for a given IP address."""
        subnet_info = self.network_info.find_subnet_for_ip("192.168.1.100")
        self.assertIsNotNone(subnet_info)
        self.assertEqual(subnet_info["device"], "ROUTER_A")

        subnet_info_none = self.network_info.find_subnet_for_ip("100.100.100.100")
        self.assertIsNone(subnet_info_none)
        
        # Test with an invalid IP format
        subnet_info_invalid = self.network_info.find_subnet_for_ip("999.999.999.999")
        self.assertIsNone(subnet_info_invalid)

    def test_find_device_by_ip(self):
        """Test finding a full device's info by an IP address within its subnet."""
        device_info = self.network_info.find_device_by_ip("172.16.50.20")
        self.assertIsNotNone(device_info)
        self.assertEqual(device_info["hostname"], "ROUTER_B")
        self.assertEqual(device_info["management_ip"], "10.0.0.2")

        device_info_none = self.network_info.find_device_by_ip("100.100.100.100")
        self.assertIsNone(device_info_none)

if __name__ == '__main__':
    unittest.main()
