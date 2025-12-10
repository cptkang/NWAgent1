import unittest
from tools.network_tools import ip_in_subnet

class TestNetworkTools(unittest.TestCase):

    def test_ip_in_subnet(self):
        """ip_in_subnet 함수가 정상적으로 동작하는지 테스트합니다."""
        self.assertTrue(ip_in_subnet("192.168.1.50", "192.168.1.0/24"))
        self.assertFalse(ip_in_subnet("192.168.2.50", "192.168.1.0/24"))
        self.assertTrue(ip_in_subnet("10.0.0.1", "10.0.0.0/30"))
        self.assertFalse(ip_in_subnet("10.0.0.5", "10.0.0.0/30"))

if __name__ == '__main__':
    unittest.main()