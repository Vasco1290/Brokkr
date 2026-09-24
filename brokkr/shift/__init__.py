"""Image corruptions that simulate bad real-world camera conditions.

Self-contained on purpose: this package is shared with the Argos project, so it imports
nothing from the rest of Brokkr and needs only NumPy and Pillow.
"""

from .corruptions import CORRUPTIONS, SEVERITIES, corrupt

__all__ = ["CORRUPTIONS", "SEVERITIES", "corrupt"]
