from datetime import datetime
from enum import Enum
from pathlib import Path


DASH = '-' * 55
RESFOLDER_DEFAULT = "%Y-%m-%d_%H-%M-%S"
SEP = "/"
RES_FILENAME = "res.pkl"
DAG_FILENAME = "dag.png"
TSDAG_FILENAME = "ts_dag.png"

def bold(msg):
    """
    Adds bold font option to the msg 

    Args:
        msg (str): message to make bold

    Returns:
        str: bold message
    """
    return '\033[1m' + msg + '\033[0m'


def create_resfolder(resfolder):
    """
    Creates resfolder if doesn't exist

    Args:
        resfolder (str): result folder
    """
    if resfolder is None:
        now = datetime.now()

        resfolder = now.strftime(RESFOLDER_DEFAULT)

    Path(resfolder).mkdir(parents=True, exist_ok=True)

    return SEP.join([resfolder, RES_FILENAME]), SEP.join([resfolder, DAG_FILENAME]), SEP.join([resfolder, TSDAG_FILENAME])


class Thres(Enum):
    INIT = -1
    NOFOUND = None