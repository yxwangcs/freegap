import numpy as np
import logging
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


def evaluate_gap_k_noisy_max():
    k = 5
    logger.info('Evaluating Gap K Noisy Max')
    epsilon = 0.7
    input_data = np.array([2.46548342e+01, -3.28520123e+01, 5.59559541e+01, -1.32241160e+02,
                           8.79922887e+00, -5.75511548e-02, -2.93453806e+01, -4.95746889e+01,
                           -7.19644328e+01, 1.46665215e+01])

    result_1, result_2 = [], []
    for _ in range(100):
        indices, gaps = gap_k_noisy_max(input_data, k, 0.5 * epsilon)
        estimates = laplace_mechanism(input_data, indices, 0.5 * epsilon)
        truth = input_data[indices]

        coefficient = np.eye(k, k) * 3 + np.eye(k, k, 1) * -1 + np.eye(k, k, -1) * -1
        coefficient[0][0] = 2
        coefficient[k - 1][k - 1] = 2
        print(coefficient)
        b = np.append(gaps, 0)
        b = b - np.roll(b, 1) + estimates
        final_estimates = np.linalg.solve(coefficient, b)
        final_estimates = 0.25 * final_estimates + 0.75 * estimates
        result_1.append(np.sum(np.square((final_estimates - truth))) / k)

        # independently rerun naive approach
        indices, _ = gap_k_noisy_max(input_data, k, 0.5 * epsilon)
        estimates = laplace_mechanism(input_data, indices, 0.5 * epsilon)
        result_2.append(np.sum(np.square(estimates - truth)) / k)

    result_1 = np.asarray(result_1)
    result_2 = np.asarray(result_2)
    print('Refine estimate error: {} with error: {} and {}'.format(result_1.mean(), result_1.max() - result_1.mean(), result_1.mean() - result_1.min()))
    print('Naive estimate error: {} with error: {} and {}'.format(result_2.mean(), result_2.max() - result_2.mean(),
                                                                   result_2.mean() - result_2.min()))

    """
    print('gaps {}'.format(gaps))
    print('truth: {}'.format(truth))
    print('Laplace estimates: {}'.format(estimates))
    print('Our estimates: {}'.format(final_estimates))
    print('our error: {}'.format(np.sum((final_estimates - truth) * (final_estimates - truth)) / k))
    print('Direct estimate error: {}'.format(np.sum((estimates - truth) * (estimates - truth)) / k))
    """


