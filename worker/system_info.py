"""
System information utilities for worker nodes.

Provides functions to gather node identity and hardware state
for inclusion in heartbeat payloads sent to the orchestrator.
All functions are designed to work on both Raspberry Pi OS and
standard Linux development machines.
"""

import platform
import socket

import psutil

# Interface name prefixes used to distinguish wired from wireless links
# on Linux (both legacy "ethX"/"wlanX" and systemd predictable names
# like "enp3s0"/"wlp2s0"). Not meaningful on other platforms, but this
# module is Pi/Linux-focused per its own scope.
_ETHERNET_PREFIXES = ("eth", "en")


def get_node_id():
    """
    Return a stable unique identifier for this node.

    Uses the hostname, which is stable across reboots on
    Raspberry Pi OS and most Linux distributions.

    Returns
    -------
    str
        The system hostname.
    """
    return socket.gethostname()


def get_ip_address(override: str | None = None) -> str:
    """
    Return the node's primary IP address.

    Prefers the address of a wired (Ethernet) interface when one is
    present, since a machine with both Ethernet and Wi-Fi active will
    often have its default internet route on Wi-Fi even when Ethernet
    is the intended link to the rest of the cluster - a plain
    "connect out and see what IP that picks" trick reports whichever
    interface reaches the internet, not whichever one is actually on
    the cluster's LAN. Falls back to that trick only if no wired
    interface with an IPv4 address is found.

    Parameters
    ----------
    override : str or None
        If given, this address is returned as-is and nothing is
        auto-detected. Lets a user pin the reported IP explicitly
        (e.g. via a CLI flag) when detection picks the wrong link.

    Returns
    -------
    str
        The node's IPv4 address as a dotted-quad string.
    """
    if override:
        return override

    for name, addrs in psutil.net_if_addrs().items():
        if not name.startswith(_ETHERNET_PREFIXES):
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET:
                return addr.address

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def get_available_ram_mb():
    """
    Return currently available RAM in megabytes.

    Uses psutil to report memory that can be allocated to processes
    without swapping. This includes free memory plus reclaimable
    buffers and cache.

    Returns
    -------
    int
        Available RAM in megabytes.
    """
    return int(psutil.virtual_memory().available // (1024 * 1024))


def get_total_ram_mb():
    """
    Return total physical RAM in megabytes.

    Returns
    -------
    int
        Total RAM in megabytes.
    """
    return int(psutil.virtual_memory().total // (1024 * 1024))


def get_cpu_count():
    """
    Return the number of logical CPU cores.

    Returns
    -------
    int
        Number of logical CPU cores.
    """
    return psutil.cpu_count(logical=True)


def get_arch():
    """
    Return the CPU architecture string.

    Returns
    -------
    str
        Architecture identifier, e.g. "aarch64" on Raspberry Pi 5,
        "x86_64" on a development laptop.
    """
    return platform.machine()


def get_os_name():
    """
    Return the operating system name.

    Returns
    -------
    str
        OS name, e.g. "Linux".
    """
    return platform.system()


def get_os_version():
    """
    Return the OS kernel version string.

    Returns
    -------
    str
        Kernel version, e.g. "6.6.31+rpt-rpi-2712".
    """
    return platform.release()
