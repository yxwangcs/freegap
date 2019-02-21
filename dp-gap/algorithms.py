import numpy as np


def sparse_vector_technique(q, threshold, c, epsilon):
    out = []
    count = 0
    i = 1
    eta = np.random.laplace(2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i <= len(q) and count < c:
        eta_i = np.random.laplace(4.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(True)
            count += 1
        else:
            out.append(False)
    return out


def sparse_vector_technique(q, threshold, c, epsilon):
    out = []
    count = 0
    i = 1
    eta = np.random.laplace(2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i <= len(q) and count < c:
        eta_i = np.random.laplace(4.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(noisy_q_i - noisy_threshold)
            count += 1
        else:
            out.append(False)
    return out


def gap_noisy_max(q, epsilon):
    i, imax, max, gap = 1, 1, 0, 0
    while i <= len(q):
        eta_i = np.random.laplace(2.0 / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i > max or i == 1:
            imax = i
            gap = noisy_q_i - max
            max = noisy_q_i
        else:
            if noisy_q_i > max - gap:
                gap = max - noisy_q_i
    return imax, gap
