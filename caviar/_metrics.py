# Author: Lee Yat Shun, Jasper
# Copyright (c) 2023 Lee Yat Shun, Jasper. All rights reserved.

import numpy as np

def hit(returns, vars, quantile):
    """
    :params: returns (array):
    :params: vars (array):
    :params: quantile (float):
    returns: hit (%)
    """
    return sum(np.where(returns*100 <= vars, 1, 0))/returns.shape[0])