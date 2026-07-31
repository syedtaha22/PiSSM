"""
Tests for system info utilities.

Validates that each function returns the correct type and a
reasonable value. These functions wrap platform and psutil calls
to report node identity and hardware state for heartbeat payloads.
"""

import socket
from unittest.mock import patch

from worker.system_info import (
    get_arch,
    get_available_ram_mb,
    get_cpu_count,
    get_ip_address,
    get_node_id,
    get_os_name,
    get_os_version,
    get_total_ram_mb,
)


def _snicaddr(family, address):
    """
    Build a stand-in for psutil's snicaddr namedtuple.

    Parameters
    ----------
    family : socket.AddressFamily
        Address family, e.g. socket.AF_INET.
    address : str
        The address string.

    Returns
    -------
    object
        An object with .family and .address attributes, matching the
        subset of snicaddr's interface get_ip_address actually reads.
    """
    return type("snicaddr", (), {"family": family, "address": address})()


class TestNodeIdentity:
    """
    Tests for node identification functions.
    """

    def test_get_node_id_returns_nonempty_string(self):
        """
        Node ID must be a non-empty string.
        """
        node_id = get_node_id()
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_get_ip_address_returns_valid_ipv4(self):
        """
        IP address must be a dotted-quad IPv4 string with four octets.
        """
        ip = get_ip_address()
        assert isinstance(ip, str)
        parts = ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert part.isdigit()
            assert 0 <= int(part) <= 255


class TestIpAddressDetection:
    """
    Tests for get_ip_address's Ethernet-over-Wi-Fi preference and
    override support.
    """

    def test_override_is_returned_as_is(self):
        """
        An explicit override skips detection entirely.
        """
        assert get_ip_address("10.0.0.5") == "10.0.0.5"

    def test_prefers_ethernet_interface_over_wifi(self):
        """
        A wired interface's address wins even if Wi-Fi is also up -
        this is the actual bug: connecting out to detect the default
        route often picks Wi-Fi even when Ethernet is the intended
        cluster link.
        """
        fake_interfaces = {
            "lo": [_snicaddr(socket.AF_INET, "127.0.0.1")],
            "wlan0": [_snicaddr(socket.AF_INET, "192.168.1.50")],
            "eth0": [_snicaddr(socket.AF_INET, "192.168.1.99")],
        }
        with patch(
            "worker.system_info.psutil.net_if_addrs", return_value=fake_interfaces
        ):
            assert get_ip_address() == "192.168.1.99"

    def test_falls_back_to_default_route_when_no_wired_interface(self):
        """
        With no eth*/en* interface present, falls back to the
        connect-out trick rather than returning nothing.
        """
        fake_interfaces = {
            "lo": [_snicaddr(socket.AF_INET, "127.0.0.1")],
            "wlan0": [_snicaddr(socket.AF_INET, "192.168.1.50")],
        }
        with patch(
            "worker.system_info.psutil.net_if_addrs", return_value=fake_interfaces
        ):
            ip = get_ip_address()
        assert isinstance(ip, str)
        assert len(ip.split(".")) == 4


class TestMemoryInfo:
    """
    Tests for RAM reporting functions.
    """

    def test_get_available_ram_mb_returns_positive_int(self):
        """
        Available RAM must be a positive integer in megabytes.
        """
        ram = get_available_ram_mb()
        assert isinstance(ram, int)
        assert ram > 0

    def test_get_total_ram_mb_returns_positive_int(self):
        """
        Total RAM must be a positive integer in megabytes.
        """
        ram = get_total_ram_mb()
        assert isinstance(ram, int)
        assert ram > 0

    def test_available_ram_does_not_exceed_total(self):
        """
        Available RAM must not exceed total RAM.
        """
        available = get_available_ram_mb()
        total = get_total_ram_mb()
        assert available <= total


class TestHardwareInfo:
    """
    Tests for CPU and architecture reporting.
    """

    def test_get_cpu_count_returns_positive_int(self):
        """
        CPU count must be a positive integer.
        """
        count = get_cpu_count()
        assert isinstance(count, int)
        assert count > 0

    def test_get_arch_returns_nonempty_string(self):
        """
        Architecture must be a non-empty string like "x86_64" or "aarch64".
        """
        arch = get_arch()
        assert isinstance(arch, str)
        assert len(arch) > 0


class TestOSInfo:
    """
    Tests for operating system reporting.
    """

    def test_get_os_name_returns_nonempty_string(self):
        """
        OS name must be a non-empty string like "Linux".
        """
        name = get_os_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_os_version_returns_nonempty_string(self):
        """
        OS version must be a non-empty string (kernel version).
        """
        version = get_os_version()
        assert isinstance(version, str)
        assert len(version) > 0
