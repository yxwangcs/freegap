import os
import numpy as np
import matplotlib.pyplot as plt


def _lapprod(a, b, epsa, epsb, prng=np.random):
    assert a >= b, "a>=b required for lapprod"
    pow_first = np.exp(np.float128((epsb - epsa) * abs(a - b)))
    pow_second = np.exp(np.float128((epsa - epsb) * abs(a - b)))
    A = B = C = 0.0
    if epsa >= epsb:
        A = 1.0 / float(epsb + epsa)
        B = (1.0 - pow_first) / float(epsa - epsb)
        C = pow_first / float(epsa + epsb)
    else:
        A = pow_second / float(epsb + epsa)
        B = (pow_second - 1.0) / float(epsa - epsb)
        C = 1.0 / float(epsa + epsb)
    U = prng.uniform(0, A + B + C)
    X = 0.0

    if U <= A:
        E = prng.exponential(1.0 / float(epsa + epsb))
        X = a + E
    elif U <= A + C:
        E = prng.exponential(1.0 / float(epsa + epsb))
        X = b - E
    else:
        E = prng.exponential(1.0 / abs(epsa - epsb))
        F = E % (a - b)
        if epsa > epsb:
            X = a - F
        else:
            X = b + F
    return X


def refinelaplace(y, epsprime, eps, mu=0, prng=np.random):
    assert epsprime > eps, "epsprime must be bigger than eps"
    sample_uniform = prng.uniform(0, 1)
    bound = eps / float(epsprime) * np.exp((eps - epsprime) * abs(y - mu))

    if sample_uniform <= bound:
        return y
    elif y >= mu:
        return _lapprod(y, mu, eps, epsprime, prng)
    else:
        return _lapprod(mu, y, epsprime, eps, prng)


def evaluate_refine_laplace(output_folder='./figures/refinelap'):
    # create the output folder if not exists
    try:
        os.makedirs(output_folder)
    except FileExistsError:
        pass
    path_prefix = os.path.abspath(output_folder)

    loc, scale, refined_scale = 0, 1, 1 / 2.0
    s = np.random.laplace(loc, scale=scale, size=10000)
    _ = plt.hist(s, 150, density=True, range=(-4., 4))
    x = np.arange(-4., 4., .01)
    pdf = np.exp(-abs(x - loc) / scale) / (2. * scale)
    plt.plot(x, pdf, label='\\huge Laplace($\\mu$=0, scale=1)', linewidth=3)
    plt.title('\\huge X: Laplace')
    axes = plt.gca()
    axes.set_ylim([0., 1.])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.savefig('{}/lap.pdf'.format(path_prefix))
    plt.clf()

    # plot refined laplace
    s = np.fromiter((refinelaplace(elem, 2, 1) for elem in s), dtype=np.float)
    _ = plt.hist(s, 150, density=True, range=(-4., 4))
    refined_pdf = np.exp(-abs(x - loc) / refined_scale) / (2. * refined_scale)
    plt.plot(x, refined_pdf, label='\\huge Laplace($\\mu$=0, scale={})'.format(refined_scale), linewidth=3)
    plt.title('\\huge RefineLap (X, 1, 2)', fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.savefig('{}/refinelap.pdf'.format(path_prefix))
    plt.clf()
