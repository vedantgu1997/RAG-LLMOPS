import os
import fitz
import uuid
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentHandler:
    """
    Handles PDF saving and reading operations.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self, data_dir = None, session_id = None):
        self.log = CustomLogger().get_logger(__name__)
        self.data_dir = data_dir or os.getenv(
            "DATA_STRORAGE_PATH",
            os.path.join(os.getcwd(), "data", "document_analysis")
        )
        self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Create base session directory
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)

        self.log.info("PDFHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, pdf_data):
        try:
            pdf_path = os.path.join(self.data_dir, f"{self.session_id}.pdf")
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_data)
            self.log.info("PDF saved successfully", pdf_path=pdf_path)
        except Exception as e:
            self.log.error(f"Error saving PDF: {str(e)}")
            raise DocumentPortalException("Failed to save PDF}", e) from e #type: ignore

    def read_pdf(self, pdf_path):
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1): #type: ignore
                    text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
                text = "\n".join(text_chunks)

                self.log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id)
            return text
        except Exception as e:
            self.log.error(f"Error reading PDF: {str(e)}")
            raise DocumentPortalException("Failed to read PDF", e) from e #type: ignore