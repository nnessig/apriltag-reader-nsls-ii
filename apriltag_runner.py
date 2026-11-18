#!/usr/bin/env python3
"""Compatibility wrapper: ``python3 apriltag_runner.py`` runs ``apriltag_reader.cli.main()``."""

import os
import sys

try:
    from apriltag_reader.cli import main
except ImportError:
    # Not installed: fall back to the src/ layout next to this file
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from apriltag_reader.cli import main

if __name__ == "__main__":
    main()
