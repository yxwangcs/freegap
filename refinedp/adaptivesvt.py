import logging
import numpy as np
from refinedp.refinelaplace import refinelaplace


logger = logging.getLogger(__name__)


def adaptive_sparse_vector(q, epsilon, c, threshold):
    out = []
    count = 1
    refine_count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < 2 * c - 1:
        eta_i = np.random.laplace(scale=8.0 * c / epsilon)
        xi_i = refinelaplace(eta_i, epsilon / (4.0 * c), epsilon / (8.0 * c))
        if q[i] + eta_i >= noisy_threshold + 16.0 * np.sqrt(2) * c / epsilon:
            out.append(True)
            count += 1
        elif q[i] + xi_i > noisy_threshold:
            refine_count += 1
            out.append(True)
            count += 2
        else:
            out.append(False)
        i += 1
    logger.debug('Total refined: {}'.format(refine_count))

    return np.nonzero(out), None


def sparse_vector(q, epsilon, c, threshold):
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < c:
        eta_i = np.random.laplace(scale=4.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(True)
            count += 1
        else:
            out.append(False)
        i += 1
    return np.nonzero(out), None
