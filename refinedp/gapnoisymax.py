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


def goal(x, C, estimates, gaps):
    #print('x: {} estimates: {}, gaps: {}'.format(x, estimates, gaps))
    #copy = np.array(x, copy=True)
    to_solve = np.concatenate((x - estimates, x[:-1] - x[1:] - gaps))
    #print(to_solve)
    to_solve.shape = 1, len(to_solve)
    transposed = np.transpose(to_solve)
    print(np.matmul(np.matmul(to_solve, C), transposed)[0][0])
    return np.matmul(np.matmul(to_solve, C), transposed)[0][0]


def gap_max_estimates(q, epsilon, k):
    indices, gaps = gap_k_noisy_max(q, 0.5 * epsilon, k)
    estimates = laplace_mechanism(q, 0.5 * epsilon, indices)
    """ old method, this one is asymmetric
    coefficient = np.eye(k, k) * 10 + np.eye(k, k, 1) * -1 + np.eye(k, k, -1) * -1
    coefficient[0][0] = 9
    coefficient[k - 1][k - 1] = 9
    b = np.append(gaps, 0)
    b = b - np.roll(b, 1) + 8 * estimates
    final_estimates = np.linalg.solve(coefficient, b)
    """
    """ `new` method
    A = np.full((k, k), -1) + np.eye(k, k) * (k + 7 + 1)
    B = np.vstack([np.fromiter((k - (i + 1) for i in range(k - 1)), dtype=np.float, count=k - 1) for _ in range(k)])
    C = np.vstack([np.fromiter((k if j < i else 0 for j in range(k - 1)), dtype=np.float) for i in range(k)])
    B = B - C
    estimates.shape = len(estimates), 1
    gaps.shape = len(gaps), 1
    right = 8 * estimates + np.matmul(B, gaps)
    final_estimates = np.linalg.solve(A, right)
    final_estimates = np.squeeze(np.asarray(final_estimates))
    if k == 2:
        assert np.fabs(final_estimates[0] - (9 * estimates[0] + estimates[1] + gaps[0]) / 10) <= 0.00005
        assert np.fabs(final_estimates[1] - (estimates[0] + 9 * estimates[1] - gaps[0]) / 10) <= 0.00005
    elif k == 3:
        x1 = (9 * estimates[0] + estimates[1] + estimates[2] + gaps[0] + gaps[0] + gaps[1]) / 11
        x2 = (estimates[0] + 9 * estimates[1] + estimates[2] - gaps[0] + gaps[1]) / 11
        x3 = (estimates[0] + estimates[1] + 9 * estimates[2] - (gaps[0] + gaps[1]) - gaps[1]) / 11
        assert np.fabs(final_estimates[0] - x1) <= 0.00005
        assert np.fabs(final_estimates[1] - x2) <= 0.00005
        assert np.fabs(final_estimates[2] - x3) <= 0.00005
    """
    A = np.hstack((np.eye(k, k), np.zeros((k, k - 1))))
    B = np.hstack((np.zeros((k - 1, k)), np.eye(k - 1, k - 1) * 8 + np.eye(k - 1, k - 1, -1) * -4 + np.eye(k - 1, k - 1, 1) * -4))
    C = np.vstack((A, B))
    from scipy.optimize import minimize
    final_estimates = minimize(goal, estimates, args=(np.linalg.inv(C), estimates, gaps))
    final_estimates = final_estimates.x if final_estimates.success else None
    return indices, final_estimates
