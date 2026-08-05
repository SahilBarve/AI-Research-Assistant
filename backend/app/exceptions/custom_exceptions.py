"""
Custom exception classes for the AI Research Assistant.

These exceptions provide meaningful error types that can be
handled consistently across the application.
"""


class AIResearchAssistantException(Exception): # Python only treats subclasses of Exception as exceptions that can be raised and caught.
    """
    Base exception for the application.

    All custom exceptions should inherit from this class.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class DocumentTooLargeException(AIResearchAssistantException):
    """
    Raised when an uploaded document exceeds
    the allowed file size.
    """

    pass


class InvalidFileTypeException(AIResearchAssistantException):
    """
    Raised when the uploaded file type
    is not supported.
    """

    pass


class DocumentNotFoundException(AIResearchAssistantException):
    """
    Raised when a requested document
    cannot be found.
    """

    pass


class InvalidDocumentTypeException(AIResearchAssistantException):
    def __init__(self):
        super().__init__(
            message="Only PDF files are allowed.",
           
        )

class DuplicateDocumentException(AIResearchAssistantException):
    """
    Raised when a document with the same name
    already exists.
    """

    def __init__(self):
        super().__init__(
            "A document with this filename already exists."
        )