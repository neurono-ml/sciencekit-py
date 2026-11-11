"""Structured exception hierarchy mapped to `SKFfiStatus` codes."""

from __future__ import annotations


class ScienceKitError(Exception):
    """Base exception carrying the stable `sciencekit_ffi` status code and message."""

    status_code: int
    status_name: str
    message: str

    def __init__(self, status_code: int, status_name: str, message: str) -> None:
        self.status_code = int(status_code)
        self.status_name = str(status_name)
        self.message = str(message)
        super().__init__(f"sciencekit {self.status_name} ({self.status_code}): {self.message}")


class InvalidHandleError(ScienceKitError, RuntimeError):
    """Raised when an operation uses a closed or invalid handle (`STATUS_INVALID_HANDLE = 1`)."""


class ShapeError(ScienceKitError, ValueError):
    """Raised when input shapes or dimensions are incompatible (`STATUS_SHAPE = 2`)."""


class InvalidArgumentError(ScienceKitError, ValueError):
    """Raised when an argument, dtype, rank, or path is malformed (`STATUS_INVALID_ARGUMENT = 3`)."""


class UnsupportedError(ScienceKitError, NotImplementedError):
    """Raised when an operation or layout is unsupported (`STATUS_UNSUPPORTED = 4`)."""


class NotFittedError(ScienceKitError, RuntimeError):
    """Raised when an operation requires a fitted model (`STATUS_NOT_FITTED = 5`)."""


class IoError(ScienceKitError, OSError):
    """Raised when saving or loading a model fails at the filesystem (`STATUS_IO = 6`)."""


class InternalError(ScienceKitError, RuntimeError):
    """Raised on an unexpected internal failure or caught panic (`STATUS_INTERNAL = 7`)."""


STATUS_EXCEPTION_MAP: dict[int, type[ScienceKitError]] = {
    1: InvalidHandleError,
    2: ShapeError,
    3: InvalidArgumentError,
    4: UnsupportedError,
    5: NotFittedError,
    6: IoError,
    7: InternalError,
}
