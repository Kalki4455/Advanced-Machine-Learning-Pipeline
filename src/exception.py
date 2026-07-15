import traceback

class CustomException(Exception):

    def __init__(self, error):

        self.error = error

        self.trace = traceback.format_exc()

        super().__init__(str(error))

    def __str__(self):

        return f"{self.error}\n{self.trace}"