"""Lets the scripts import the skinfw package when run from the repository root."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
