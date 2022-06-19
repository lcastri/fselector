from enum import Enum


DASH = '-' * 55

def bold(msg):
    return '\033[1m' + msg + '\033[0m'

class Thres(Enum):
    INIT = -1
    NOFOUND = None