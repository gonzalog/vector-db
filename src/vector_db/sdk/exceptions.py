"""Exceptions for the Vector DB SDK."""


class VectorDBSDKError(Exception):
    """Base exception for Vector DB SDK errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class NotFoundError(VectorDBSDKError):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class AlreadyExistsError(VectorDBSDKError):
    """Raised when a resource already exists (409)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=409)


class ValidationError(VectorDBSDKError):
    """Raised when validation fails (422)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class ServerError(VectorDBSDKError):
    """Raised when the server returns a 5xx error."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code=status_code)


class ConnectionError(VectorDBSDKError):
    """Raised when connection to the API fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=None)
