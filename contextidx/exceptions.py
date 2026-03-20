"""Typed exception hierarchy for contextidx.

All public exceptions inherit from :class:`ContextIdxError` so callers
can catch a single base class or handle specific failure modes.
"""

from __future__ import annotations


class ContextIdxError(Exception):
    """Base exception for all contextidx errors."""


class ConfigurationError(ContextIdxError):
    """Raised when a constructor parameter or configuration value is invalid."""


class StoreError(ContextIdxError):
    """Raised when an internal metadata store operation fails."""


class BackendError(ContextIdxError):
    """Raised when a vector backend operation fails."""


class EmbeddingError(ContextIdxError):
    """Raised when embedding generation fails (API error, rate limit, etc.)."""


class ConflictError(ContextIdxError):
    """Raised when conflict resolution encounters an unrecoverable issue."""
