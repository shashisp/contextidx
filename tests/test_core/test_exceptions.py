"""Tests for the exception hierarchy."""

import pytest

from contextidx.exceptions import (
    BackendError,
    ConfigurationError,
    ContextIdxError,
    ConflictError,
    EmbeddingError,
    StoreError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for exc_cls in (
            ConfigurationError,
            StoreError,
            BackendError,
            EmbeddingError,
            ConflictError,
        ):
            assert issubclass(exc_cls, ContextIdxError)

    def test_catch_base_catches_all(self):
        for exc_cls in (
            ConfigurationError,
            StoreError,
            BackendError,
            EmbeddingError,
            ConflictError,
        ):
            with pytest.raises(ContextIdxError):
                raise exc_cls("test")

    def test_specific_catch(self):
        with pytest.raises(StoreError):
            raise StoreError("db error")

        with pytest.raises(BackendError):
            raise BackendError("vector db error")

    def test_does_not_catch_unrelated(self):
        with pytest.raises(ValueError):
            raise ValueError("unrelated")

    def test_message_preserved(self):
        exc = EmbeddingError("rate limit exceeded")
        assert str(exc) == "rate limit exceeded"

    def test_chaining(self):
        original = ConnectionError("network down")
        try:
            raise BackendError("Backend failed") from original
        except BackendError as wrapped:
            assert wrapped.__cause__ is original
