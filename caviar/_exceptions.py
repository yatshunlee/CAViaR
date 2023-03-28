class InputSizeError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"{self.message}"
    
class NotFittedError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"{self.message}"
        
class ConvergenceError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"{self.message}"