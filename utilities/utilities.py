from datetime import datetime
from enum import Enum
from pathlib import Path


DASH = '-' * 55
RESFOLDER_DEFAULT = "%Y-%m-%d_%H-%M-%S"
SEP = "/"
RES_FILENAME = "res.pkl"
DAG_FILENAME = "dag.png"
TSDAG_FILENAME = "ts_dag.png"
DEP_FILENAME = "dependency.png"
LOG_FILENAME = "log.txt"
RESULTS_FILENAME = "results"

def bold(msg):
    """
    Adds bold font option to the msg 

    Args:
        msg (str): message to make bold

    Returns:
        str: bold message
    """
    return '\033[1m' + msg + '\033[0m'


def get_selectorpath(resfolder):
    """
    Return log file path

    Args:
        resfolder (str): result folder

    Returns:
        (str, str): log file path, dependency image file path
    """
    Path(SEP.join([RESULTS_FILENAME, resfolder])).mkdir(parents=True, exist_ok=True)
    return SEP.join([RESULTS_FILENAME, resfolder, LOG_FILENAME]), SEP.join([RESULTS_FILENAME, resfolder, DEP_FILENAME])


def create_results_folder():
    """
    Creates resutls if doesn't exist
    """
    Path(RESULTS_FILENAME).mkdir(parents=True, exist_ok=True)


def get_validatorpaths(resfolder):
    """
    Creates resfolder if doesn't exist

    Args:
        resfolder (str): result folder name

    Returns:
        (str, str, str): result.pkl file path, dag file path, ts_dag file path
    """
    Path(SEP.join([RESULTS_FILENAME, resfolder])).mkdir(parents=True, exist_ok=True)
    return SEP.join([RESULTS_FILENAME, resfolder, RES_FILENAME]), SEP.join([RESULTS_FILENAME, resfolder, DAG_FILENAME]), SEP.join([RESULTS_FILENAME, resfolder, TSDAG_FILENAME])


class Thres(Enum):
    INIT = -1
    NOFOUND = None