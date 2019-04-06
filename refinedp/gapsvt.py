import logging
import numpy as np
from refinedp.algorithms import sparse_vector, laplace_mechanism


logger = logging.getLogger(__name__)


def gap_sparse_vector(q, epsilon, k, threshold, allocation=(0.5, 0.5)):
    x, y = allocation
    assert abs(x + y - 1.0) < 1e-05
    epsilon_1, epsilon_2 = x * epsilon, y * epsilon
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=1.0 / epsilon_1)
    noisy_threshold = threshold + eta
    while i < len(q) and count < k:
        eta_i = np.random.laplace(scale=2.0 * k / epsilon_2)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(noisy_q_i - noisy_threshold)
            count += 1
        else:
            out.append(False)
        i += 1
    return out


def gap_svt_estimates(q, epsilon, k, threshold):
    # budget allocation for gap svt
    x, y = 1, np.power(2 * k, 2.0 / 3.0)
    gap_x, gap_y = x / (x + y), y / (x + y)

    # budget allocation between gap / laplace
    x, y = 1, 1
    gap_budget, lap_budget = y / (x + y), x / (x + y)

    answers = np.asarray(gap_sparse_vector(q, gap_budget * epsilon, k, threshold, allocation=(gap_x, gap_y)))
    indices = np.nonzero(answers)[0]
    initial_estimates = answers[indices] + threshold
    direct_estimates = laplace_mechanism(q, lap_budget * epsilon, indices)
    variance_gap = 8 * np.power((1 + np.power(2 * k, 2.0 / 3)), 3) / np.square(epsilon)
    variance_lap = 8 * np.square(k) / np.square(epsilon)

    # variance_lap = 2.0 * c * c / ((epsilon * lap_budget) * (epsilon * lap_budget))
    # variance_gap = (32 + 128 * np.square(c)) / np.square(epsilon)

    # do weighted average
    return indices, (initial_estimates / variance_gap + direct_estimates / variance_lap) / (1.0 / variance_gap + 1.0 / variance_lap)


def svt_baseline_estimates(q, epsilon, k, threshold):
    x, y = 1, np.power(2 * k, 2.0 / 3.0)
    gap_x, gap_y = x / (x + y), y / (x + y)
    answers = sparse_vector(q, epsilon / 2.0, k, threshold, allocation=(gap_x, gap_y))
    return np.nonzero(answers)[0], np.asarray(laplace_mechanism(q, epsilon / 2.0, np.nonzero(answers)))
