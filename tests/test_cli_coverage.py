"""
CLI module tests for coverage increase.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# CLI fonksiyonlarını import etmeye çalış
try:
    from addrnorm.cli import create_parser, main, process_single_address

    CLI_AVAILABLE = True
except ImportError as e:
    CLI_AVAILABLE = False
    import_error = str(e)


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
def test_create_parser():
    """Test argument parser creation."""
    parser = create_parser()
    assert parser is not None

    # Test with sample args
    args = parser.parse_args(["--address", "İstanbul Beşiktaş"])
    assert args.address == "İstanbul Beşiktaş"


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
def test_process_single_address():
    """Test single address processing."""
    result = process_single_address(
        address="İstanbul Beşiktaş", method="pattern_high", return_components=False
    )
    assert isinstance(result, dict)
    assert "normalized_address" in result


@patch("sys.argv", ["addrnorm", "--address", "İstanbul Beşiktaş"])
def test_main_single_address():
    """Test main function with single address."""
    with patch("addrnorm.cli.process_single_address") as mock_process:
        mock_process.return_value = {
            "normalized_address": "İstanbul Beşiktaş",
            "success": True,
        }
        try:
            main()
        except SystemExit:
            pass  # Expected for CLI tools
        mock_process.assert_called_once()


@patch("sys.argv", ["addrnorm", "--help"])
def test_main_help():
    """Test help functionality."""
    with pytest.raises(SystemExit):
        main()


def test_cli_with_different_methods():
    """Test CLI with different normalization methods."""
    methods = ["pattern_high", "pattern_medium", "pattern_low"]

    for method in methods:
        result = process_single_address(
            address="Ankara Çankaya", method=method, return_components=False
        )
        assert isinstance(result, dict)
        assert "normalized_address" in result


def test_cli_with_components():
    """Test CLI with component return."""
    result = process_single_address(
        address="İstanbul Beşiktaş Levent",
        method="pattern_high",
        return_components=True,
    )
    assert isinstance(result, dict)
    assert "normalized_address" in result
