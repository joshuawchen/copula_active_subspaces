"""Pytest configuration."""
import os
import sys

# Make the package importable without requiring `pip install -e .`
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
