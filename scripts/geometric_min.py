import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

def var(theta, eps, k):
    return np.exp(theta*eps) / ((np.exp(theta*eps) - 1)**2) + np.exp((1-theta)*eps/k) / ((np.exp((1-theta)*eps/k) - 1)**2)

x0 = 0.5
bnds = ((0.0001,0.9999),)

ks = [k for k in range(1,51)]
xs = []
for k in ks:
    res = minimize(var,
            x0,
            args=(1,k),
            method='L-BFGS-B',
            bounds=bnds
            )
    xs.append(res.x[0])

xdata = np.array(ks)
ydata = np.array(xs)

plt.plot(xdata, ydata, 'b', label=r'$\theta_{min}$ for $k$ from 1 to 50', markersize=10, marker='o', color='royalblue')


def func(k):
    return 1/(1+np.power(k, 2/3))
plt.plot(xdata, func(xdata), '-', color='orange',
         label=r'curve of $\theta = \frac{1}{1+\sqrt[3]{k^2}}$', linewidth=3)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'$k$', size=16)
plt.ylabel(r'$\theta_{min}$', size=16)
plt.legend(prop={'size':16})
plt.show()


# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# popt, pcov = curve_fit(func, xdata, ydata)
# plt.plot(xdata, func(xdata, *popt), 'r-',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

