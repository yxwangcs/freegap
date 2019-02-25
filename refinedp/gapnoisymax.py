import numpy as np


def gap_noisy_max(q, epsilon):
    i, imax, max, gap = 0, 1, 0, 0
    while i < len(q):
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


def gap_k_noisy_max(q, k, epsilon):
    assert k <= len(q), 'k must be less or equal than the length of q'
    noisy_q = np.asarray(q, dtype=np.float) + np.random.laplace(2.0 / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    return indices, gaps
