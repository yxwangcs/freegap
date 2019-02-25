import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from refinedp.refinelaplace import refinelaplace
from refinedp.algorithms import *


logger = logging.getLogger(__name__)


def adaptive_sparse_vector(q, threshold, c, epsilon):
    out = []
    count = 1
    refine_count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < 2 * c - 1:
        eta_i = np.random.laplace(scale=8.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold + 24.0 * np.sqrt(2) * c / epsilon:
            out.append(True)
            count += 1
        else:
            psi_i = refinelaplace(eta_i, 0, epsilon / (4.0 * c), epsilon / (8.0 * c))
            refine_count += 1
            noisy_q_i = q[i] + psi_i
            if noisy_q_i >= noisy_threshold:
                out.append(True)
                count += 2
            else:
                out.append(False)
        i += 1
    logger.info('Total refined: {}'.format(refine_count))

    return out

