from enum import Enum

class score_fcn(Enum):
    CCorr = "Cross-correlation"
    MI = "Mutual Information"
    HSICLasso = "HSICLasso"
    TE = "Transfer Entropy"
    MIT = "Momentary Information Transfer"

K_BEST = 'all'