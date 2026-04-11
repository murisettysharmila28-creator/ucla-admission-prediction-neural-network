import sys


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown"

        self.error_message = (
            f"Error occurred in script: [{file_name}] "
            f"at line: [{line_number}] "
            f"with message: [{str(error_message)}]"
        )

        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message