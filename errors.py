"""Errors safe to return to the browser without exposing credentials."""


class AppError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
