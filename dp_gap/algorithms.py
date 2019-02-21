import numpy as np
import matplotlib.pyplot as plt
from dp_gap.refinelap import refinelaplace



def sparse_vector_technique(q, threshold, c, epsilon):
    out = []
    count = 0
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < c:
        eta_i = np.random.laplace(scale=4.0 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold:
            out.append(True)
            count += 1
        else:
            out.append(False)
        i += 1
    return out


def gap_sparse_vector_technique(q, threshold, c, epsilon):
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


def gap_noisy_max(q, epsilon):
    i, imax, max, gap = 1, 1, 0, 0
    while i <= len(q):
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


def adaptive_sparse_vector(q, threshold, c, epsilon):
    out = []
    count = 1
    i = 0
    eta = np.random.laplace(scale=2.0 / epsilon)
    noisy_threshold = threshold + eta
    while i < len(q) and count < 2 * c - 1:
        eta_i = np.random.laplace(scale=8 * c / epsilon)
        noisy_q_i = q[i] + eta_i
        if noisy_q_i >= noisy_threshold + 24 * np.sqrt(2) * c / epsilon:
            out.append(True)
            count += 1
        else:
            psi_i = refinelaplace(eta_i, 0, 8.0 * c / epsilon, 4.0 * c / epsilon)
            print(psi_i)
            print('refined')
            noisy_q_i = q[i] + psi_i
            if noisy_q_i >= noisy_threshold:
                out.append(True)
                count += 2
            else:
                out.append(False)
        i += 1

    return out
