import numpy as np
import logging
from refinedp.refinelaplace import refinelaplace


logger = logging.getLogger(__name__)


def sparse_vector(q, threshold, c, epsilon):
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
    return out


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
        #print(noisy_q_i, noisy_threshold + 8.0 * np.sqrt(2) * c / epsilon)
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


def gap_sparse_vector(q, threshold, c, epsilon):
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < c:
        eta_i = np.random.laplace(scale=4.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(noisy_q_i - noisy_threshold)
            count += 1
        else:
            out.append(False)
        i += 1
    return out


def gap_noisy_max(q, epsilon):
    i, imax, max, gap = 1, 1, 0, 0
    while i <= len(q):
        eta_i = np.random.laplace(scale=2.0 / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i > max or i == 1:
            imax = i
            gap = noisy_q_i - max
            max = noisy_q_i
        else:
            if noisy_q_i > max - gap:
                gap = max - noisy_q_i
    return imax, gap
