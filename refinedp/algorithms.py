import numpy as np
import logging

logger = logging.getLogger(__name__)


def sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5)):
    threshold_allocation, query_allocation = allocation
    assert abs(threshold_allocation + query_allocation - 1.0) < 1e-05
    epsilon_1, epsilon_2 = threshold_allocation * epsilon, query_allocation * epsilon
    indices = []
    i, count = 0, 0
    noisy_threshold = threshold + np.random.laplace(scale=1.0 / epsilon_1)
    while i < len(q) and count < k:
        if q[i] + np.random.laplace(scale=2.0 * k / epsilon_2) >= noisy_threshold:
            indices.append(i)
            count += 1
        i += 1
    return np.asarray(indices)


def noisy_top_k(q, epsilon, k):
    assert k <= len(q), 'k must be less or equal to the length of q'
    noisy_q = q + np.random.laplace(scale=2.0 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    return indices


def laplace_mechanism(q, epsilon, indices):
    request_q = q[indices]
    return request_q + np.random.laplace(scale=float(len(request_q)) / epsilon, size=len(request_q))
