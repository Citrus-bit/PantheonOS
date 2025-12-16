"""
Context Compression module for Pantheon.

This module provides automatic context compression when conversation
history approaches the LLM's context window limit.
"""

from .compressor import (
    CompressionConfig,
    CompressionResult,
    CompressionStatus,
    ContextCompressor,
)

__all__ = [
    "CompressionConfig",
    "CompressionResult", 
    "CompressionStatus",
    "ContextCompressor",
]
