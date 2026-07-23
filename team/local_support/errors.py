"""Typed public failures shared by the local controller and HTTP adapter."""

from http import HTTPStatus


class ApiProblemError(RuntimeError):
    def __init__(self, status: HTTPStatus, message: str, *, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
