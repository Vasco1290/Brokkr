"""Stress-testing toolkit: image corruptions that simulate bad camera conditions (corruptions.py)
and measurements of whether a classifier knows when it is wrong (reliability.py).

Self-contained on purpose: this package is shared with the Argos project, so it imports
nothing from the rest of Brokkr and needs only NumPy and Pillow.
"""

from .corruptions import CORRUPTIONS, SEVERITIES, corrupt

__all__ = ["CORRUPTIONS", "SEVERITIES", "corrupt"]
