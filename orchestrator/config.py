"""
Default configuration constants for the orchestrator.

These values control heartbeat timing, failure detection thresholds,
and the default gRPC port for the orchestrator server.
"""

DEFAULT_GRPC_PORT = 50051
DEFAULT_HEARTBEAT_INTERVAL_S = 2.0
DEFAULT_MISSED_HEARTBEATS_THRESHOLD = 3
DEFAULT_REAPER_INTERVAL_S = 1.0
