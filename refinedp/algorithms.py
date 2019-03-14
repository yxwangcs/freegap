import numpy as np
import logging

logger = logging.getLogger(__name__)


def sparse_vector(q, epsilon, c, threshold, allocation=(0.5, 0.5)):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=1.0 / (epsilon * threshold_allocation))
    noisy_threshold = threshold + eta
    while i < len(q) and count < c:
        eta_i = np.random.laplace(scale=2.0 * c / (epsilon * query_allocation))
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(True)
            count += 1
        else:
            out.append(False)
        i += 1
    return out


def noisy_k_max(q, epsilon, k):
    assert k <= len(q), 'k must be less or equal to the length of q'
    noisy_q = np.asarray(q, dtype=np.float) + np.random.laplace(scale=2.0 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    return indices


def laplace_mechanism(q, epsilon, indices):
    request_q = q[indices]
    return request_q + np.random.laplace(scale=float(len(request_q)) / epsilon, size=len(request_q))
