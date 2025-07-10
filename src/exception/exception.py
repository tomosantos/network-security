import sys
from src.logging.logger import logging
import traceback

class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = NetworkSecurityException.get_detailed_error_message(
            error_message=error_message, error_detail=error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        return f"Error occured in Python script name [{file_name}] in line number [{line_number}] with the error message [{error_message}]"

    def __str__(self):
        return self.error_message