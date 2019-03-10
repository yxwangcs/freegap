import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from refinedp.preprocess import process_bms_pos
from refinedp.algorithms import laplace_mechanism


logger = logging.getLogger(__name__)


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
    noisy_q = np.asarray(q, dtype=np.float) + np.random.laplace(2.0 * k / epsilon, size=len(q))
    indices = np.argpartition(noisy_q, -k)[-k:]
    indices = indices[np.argsort(-noisy_q[indices])]
    gaps = np.fromiter((noisy_q[first] - noisy_q[second] for first, second in zip(indices[:-1], indices[1:])),
                       dtype=np.float)
    return indices, gaps


def naive_estimate(q, k, epsilon):
    # independently rerun naive approach
    indices, _ = gap_k_noisy_max(q, k, 0.5 * epsilon)
    estimates = laplace_mechanism(q, indices, 0.5 * epsilon)
    return indices, estimates


def refined_estimate(q, k, epsilon):
    indices, gaps = gap_k_noisy_max(q, k, 0.5 * epsilon)
    estimates = laplace_mechanism(q, indices, 0.5 * epsilon)
    coefficient = np.eye(k, k) * 3 + np.eye(k, k, 1) * -1 + np.eye(k, k, -1) * -1
    coefficient[0][0] = 2
    coefficient[k - 1][k - 1] = 2
    b = np.append(gaps, 0)
    b = b - np.roll(b, 1) + estimates
    final_estimates = np.linalg.solve(coefficient, b)
    return indices, final_estimates
