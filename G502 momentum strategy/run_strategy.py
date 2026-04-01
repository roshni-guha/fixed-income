#!/usr/bin/env python3
"""
Run script for G502 Multi-Horizon Momentum Strategy.

Usage:
    python run_strategy.py                    # Run with default config
    python run_strategy.py --config my.yaml   # Run with custom config
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from momentum.main import run_pipeline, main

if __name__ == "__main__":
    sys.exit(main())
