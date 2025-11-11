# ============================================================
# Voice of Address Dataset Package
# ============================================================

"""
Voice of Address dataset for HuggingFace.

This package provides utilities to create HuggingFace datasets from
Voice of Address audio data for Vietnamese administrative divisions.
"""

from .voa import VoA, create_voa_hf_dataset

__all__ = ["VoA", "create_voa_hf_dataset"]
__version__ = "1.0.0"
