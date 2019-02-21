import numpy
import numpy as np
from numpy import exp
from numpy import abs


def lapprod(a, b, epsa, epsb, prng=numpy.random):
    assert a >= b, "a>=b required for lapprod"
    powFirst = exp(np.float128((epsb - epsa) * abs(a - b)))
    powSecond = exp(np.float128((epsa - epsb) * abs(a - b)))
    A = B = C = 0.0
    if (epsa >= epsb):
        A = 1.0 / float(epsb + epsa)
        B = (1.0 - powFirst) / float(epsa - epsb)
        C = powFirst / float(epsa + epsb)
    else:
        A = powSecond / float(epsb + epsa)
        B = (powSecond - 1.0) / float(epsa - epsb)
        C = 1.0 / float(epsa + epsb)
    U = prng.uniform(0, A + B + C)
    X = 0.0
    # print C/(A+B+C), B/(A+B+C), A/(A+B+C)
    if (U <= A):
        E = prng.exponential(1.0 / float(epsa + epsb))
        X = a + E
    elif (U <= A + C):
        E = prng.exponential(1.0 / float(epsa + epsb))
        X = b - E
    else:
        E = prng.exponential(1.0 / abs(epsa - epsb))
        F = E % (a - b)
        if (epsa > epsb):
            X = a - F
        else:
            X = b + F
    return X


def refinelaplace(y, mu, epsprime, eps, prng=numpy.random):
    assert epsprime > eps, "epsprime must be bigger than eps"
    U = prng.uniform(0, 1)
    bound = eps / float(epsprime) * exp((eps - epsprime) * abs(y - mu))
    X = 0.0
    if (U <= bound):
        X = y
    elif (y >= mu):
        X = lapprod(y, mu, eps, epsprime, prng)
    else:
        X = lapprod(mu, y, epsprime, eps, prng)
    return X


def lapprod_density(x, a, b, epsa, epsb):
    A = (epsa ** 2 - epsb ** 2) / 2.0
    B = (epsa * exp(-epsb * abs(b - a)) - epsb * exp(-epsa * abs(b - a)))
    C = exp(-epsa * abs(x - a) - epsb * abs(x - b))
    return A * (C / float(B))


def lapprod_cdf(x, a, b, epsa, epsb):
    assert a >= b
    A = (epsa ** 2 - epsb ** 2) / 2.0
    B = (epsa * exp(-epsb * abs(b - a)) - epsb * exp(-epsa * abs(b - a)))
    res = 0.0
    first = 1.0 / (epsa + epsb) * exp(epsa * (b - a))
    second = 1.0 / (epsa - epsb) * (exp(epsb * (b - a)) - exp(epsa * (b - a)))
    if (x <= b):
        res = 1.0 / (epsa + epsb) * exp(epsa * (x - a) + epsb * (x - b))
    elif (x <= a):
        res = first + 1.0 / (epsa - epsb) * (exp(epsa * (x - a) + epsb * (b - x)) - exp(epsa * (b - a)))
    else:
        res = first + second + 1.0 / (epsa + epsb) * (exp(epsb * (b - a)) - exp(epsa * (a - x) + epsb * (b - x)))
    return res / float(B) * A
