
from enum import Enum

class CPLevel(Enum):
    NONE = 0
    INFO = 1 
    DEBUG = 2 


class CPrinter():
    def __init__(self):
        self.verbosity = None
    
    def set_verbosity(self, verbosity: CPLevel):
        self.verbosity = verbosity

    def info(self, msg: str):
        if self.verbosity.value >= CPLevel.INFO.value: print(msg)

    def debug(self, msg: str):
        if self.verbosity.value >= CPLevel.DEBUG.value: print(msg)

CP = CPrinter()