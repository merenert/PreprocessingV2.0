"""
CLI Module for Address Normalization

Enhanced command-line interfaces for monitoring, analytics, and system management.
"""

from .monitoring import main as monitoring_main, create_parser as monitoring_parser

__all__ = ["monitoring_main", "monitoring_parser"]
