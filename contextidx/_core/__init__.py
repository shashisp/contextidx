"""Optional Rust-accelerated hot paths with pure Python fallback.

When the compiled ``_core_rs`` extension is available (built via maturin),
the Rust implementations are used.  Otherwise, equivalent pure-Python
functions from ``_fallback`` are loaded transparently.

Check ``RUST_AVAILABLE`` to see which backend is active.
"""

try:
    from contextidx._core_rs import batch_decay, batch_score, detect_contradictions  # type: ignore[import-not-found]
    RUST_AVAILABLE = True
except ImportError:
    from contextidx._core._fallback import batch_decay, batch_score, detect_contradictions
    RUST_AVAILABLE = False

__all__ = ["batch_decay", "batch_score", "detect_contradictions", "RUST_AVAILABLE"]
