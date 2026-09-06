"""
Document processor factory.

Selects the correct processor based on
the document file extension.
"""

from pathlib import Path

from app.services.document_processors.pdf_processors import (
    PDFProcessor,
)

from app.services.document_processors.docx_processor import (
    DOCXProcessor,
)

from app.services.document_processors.txt_processor import (
    TXTProcessor,
)

from app.services.document_processors.markdown_processor import (
    MarkdownProcessor,
)


class DocumentProcessorFactory:
    """
    Factory responsible for selecting the correct
    document processor.
    """

    _processors = {

        ".pdf": PDFProcessor,

        ".docx": DOCXProcessor,

        ".txt": TXTProcessor,

        ".md": MarkdownProcessor,

        ".markdown": MarkdownProcessor,
    }

    @classmethod
    def get_processor(
        cls,
        filename: str,
    ):
        """
        Return the appropriate processor for a file.
        """

        extension = Path(
            filename
        ).suffix.lower()

        processor_class = cls._processors.get(
            extension
        )

        if processor_class is None:

            raise ValueError(
                f"Unsupported document type: {extension}"
            )

        return processor_class()