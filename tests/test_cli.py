"""
Tests for the CLI module.
"""

import json
import os
import tempfile
from io import StringIO
from unittest.mock import patch

import pytest

# Import CLI functions
from src.addrnorm.preprocess.cli import main, process_file


class TestCLI:
    """Test the CLI functionality."""

    def test_process_file_basic(self):
        """Test basic file processing functionality."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            f.write("İstanbul Kadıköy\n")
            f.write("Ankara Çankaya\n")
            input_file = f.name

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", encoding="utf-8"
            ) as f:
                output_file = f.name

            # Process the file
            result = process_file(input_file, output_file)
            assert result is True

            # Check output file
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            assert len(results) == 2
            assert results[0]["explanation_raw"] == "İstanbul Kadıköy"
            assert "istanbul" in results[0]["normalized_address"].lower()

        finally:
            # Cleanup
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_process_file_nonexistent(self):
        """Test processing non-existent file."""
        result = process_file("nonexistent_file.txt", "output.json")
        assert result is False

    def test_main_with_text_argument(self):
        """Test main function with --text argument."""
        test_args = ["cli.py", "--text", "İstanbul Kadıköy"]

        with patch("sys.argv", test_args):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with patch("sys.exit") as mock_exit:
                    main()

                    # Should not exit with error
                    mock_exit.assert_not_called()

                    # Should produce JSON output
                    output = mock_stdout.getvalue()
                    result = json.loads(output)
                    assert result["explanation_raw"] == "İstanbul Kadıköy"
                    assert "istanbul" in result["normalized_address"].lower()

    def test_main_with_stdin(self):
        """Test main function with --stdin argument."""
        test_args = ["cli.py", "--stdin"]
        test_input = "İstanbul Kadıköy\nAnkara Çankaya\n"

        with patch("sys.argv", test_args):
            with patch("sys.stdin", StringIO(test_input)):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    with patch("sys.exit") as mock_exit:
                        main()

                        # Should not exit with error
                        mock_exit.assert_not_called()

                        # Should produce JSON output
                        output = mock_stdout.getvalue()
                        results = json.loads(output)
                        assert len(results) == 2
                        assert results[0]["explanation_raw"] == "İstanbul Kadıköy"

    def test_main_with_input_file(self):
        """Test main function with --input argument."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as f:
            f.write("İstanbul Kadıköy\n")
            input_file = f.name

        try:
            test_args = ["cli.py", "--input", input_file]

            with patch("sys.argv", test_args):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    with patch("sys.exit") as mock_exit:
                        main()

                        # Should exit with code 0 (success)
                        mock_exit.assert_called_with(0)

                        # Should produce JSON output
                        output = mock_stdout.getvalue()
                        results = json.loads(output)
                        assert len(results) == 1
                        assert results[0]["explanation_raw"] == "İstanbul Kadıköy"

        finally:
            os.unlink(input_file)

    def test_main_no_arguments(self):
        """Test main function with no arguments."""
        test_args = ["cli.py"]

        with patch("sys.argv", test_args):
            with patch("sys.exit") as mock_exit:
                with patch("sys.stdout", new_callable=StringIO):
                    main()

                    # Should exit with error code 1
                    mock_exit.assert_called_with(1)

    def test_main_with_nonexistent_input_file(self):
        """Test main function with non-existent input file."""
        test_args = ["cli.py", "--input", "nonexistent.txt"]

        with patch("sys.argv", test_args):
            with patch("sys.exit") as mock_exit:
                with patch("sys.stderr", new_callable=StringIO):
                    main()

                    # Should exit with error code 1
                    mock_exit.assert_called_with(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
