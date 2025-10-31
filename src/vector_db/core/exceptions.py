"""Custom exceptions for the Vector DB application."""

from typing import Any


class VectorDBException(Exception):
    """Base exception for Vector DB."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class NotFoundException(VectorDBException):
    """Raised when a resource is not found."""

    pass


class AlreadyExistsException(VectorDBException):
    """Raised when trying to create a resource that already exists."""

    pass


class ValidationException(VectorDBException):
    """Raised when validation fails."""

    pass


class ConcurrencyException(VectorDBException):
    """Raised when a concurrency conflict occurs."""

    pass
