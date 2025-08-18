import traceback
import sys


class DocumentPortalException(Exception):
    """Base class for all exceptions in the Document Portal application."""
    def __init__(self, error_message, error_details: sys): # type: ignore
        _,_, exc_tb = sys.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename # type: ignore
        self.line_number = exc_tb.tb_lineno # type: ignore
        self.error_message = str(error_message)
        self.traceback_str = ''.join(traceback.format_exception(*error_details.exc_info()))


    def __str__(self):
        return f"""
        Error in [{self.file_name}] at line [{self.line_number}]
        Message: {self.error_message}
        Traceback: 
        {self.traceback_str}
        """
    
# Usage example
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        app_exc = DocumentPortalException(e, sys)
        # logger.error(app_exc)
        print(app_exc)





