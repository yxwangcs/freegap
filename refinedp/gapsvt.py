import numpy as np


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
