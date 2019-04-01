import numpy as np
import logging
from refinedp.algorithms import laplace_mechanism, noisy_k_max


logger = logging.getLogger(__name__)


def gap_noisy_max(q, epsilon):
    i, imax, max_val, gap = 0, 1, 0, 0
    while i < len(q):
        eta_i = np.random.laplace(scale=2.0 / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i > max_val or i == 1:
            imax = i
            gap = noisy_q_i - max_val
            max_val = noisy_q_i
        else:
            if noisy_q_i > max_val - gap:
                gap = max_val - noisy_q_i
    return imax, gap


def gap_k_noisy_max(q, epsilon, k):
    assert k <= len(q), 'k must be less or equal to the length of q'
    noisy_q = np.asarray(q, dtype=np.float) + np.random.laplace(scale=2.0 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    return indices, gaps


def max_baseline_estimates(q, epsilon, k):
    # allocate the privacy budget 1:1 to noisy k max and laplace mechanism
    indices = noisy_k_max(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    return indices, estimates


def gap_max_estimates(q, epsilon, k):
    indices, gaps = gap_k_noisy_max(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    estimates.shape = len(estimates), 1
    gaps.shape = len(gaps), 1
    X = (np.full((k, k), 1) + np.eye(k, k) * 4 * k) * (1.0 / (5 * k))
    Y = np.vstack([np.fromiter((k - (i + 1) for i in range(k - 1)), dtype=np.float, count=k - 1) for _ in range(k)])
    Y = Y - np.vstack([np.fromiter((k if j < i else 0 for j in range(k - 1)), dtype=np.float) for i in range(k)])
    Y = (1.0 / (5 * k)) * Y
    final_estimates = np.matmul(X, estimates) + np.matmul(Y, gaps)
    final_estimates = np.asarray(final_estimates.transpose()[0])
    return indices, final_estimates
