"""
Main entry point for the Turkish address normalization pipeline.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
