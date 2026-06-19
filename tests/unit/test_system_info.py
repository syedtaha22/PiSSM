"""
Tests for system info utilities.

Validates that each function returns the correct type and a
reasonable value. These functions wrap platform and psutil calls
to report node identity and hardware state for heartbeat payloads.
"""

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
