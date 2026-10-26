"""Test suite: python -m unittest discover -s tests -t .

Puts the program modules (app/), the benchmark tools and this folder on the
import path so the tests import modules by their plain names.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _folder in (os.path.join(_ROOT, "app"), os.path.join(_ROOT, "benchmark"), os.path.dirname(os.path.abspath(__file__))):
    if _folder not in sys.path:
        sys.path.insert(0, _folder)
