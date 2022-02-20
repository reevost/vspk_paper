import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
# from ripser import ripser
from persim import plot_diagrams
import time
plot_flag = False
tic = time.perf_counter()

# # random point persistence diagrams
# np.random.seed(21)  # 21 or 45
# n = 20
# random_points = np.random.random((n, 2))
# diagrams = ripser(random_points, thresh=2.0)['dgms']
# plt.figure(1)
# plt.scatter(random_points[:, 0], random_points[:, 1], c="g")
# plt.title('random point plot')
# plt.show()
# plt.figure(2)
# plot_diagrams(diagrams, show=False)
# plt.title("diagrams plot")
# plt.show()

dom = np.linspace(-0.5, 2.2, 1000)
fun = lambda x: x**4 - 3*x**3 + 2*x**2 + 1
x0, x1, x2 = 0, (9-np.sqrt(17))/8, (9+np.sqrt(17))/8
y0, y1, y2 = fun(0), fun(x1), fun(x2)

plt.figure(1)
plt.plot(dom, fun(dom))
plt.plot(dom, np.ones(1000), 'r')
plt.plot(dom, 1.52*np.ones(1000), 'r')
plt.plot(dom, 0.75*np.ones(1000), 'r')
plt.plot(dom, y1*np.ones(1000), 'r')
plt.plot(dom, y2*np.ones(1000), 'r')
plt.scatter([x0], [y0], c='k')
plt.scatter([x1], [y1], c='k')
plt.scatter([x2], [y2], c='k')
plt.show()

dom = np.linspace(0, 2.4, 1000)
plt.figure(2)
plt.plot(dom, dom, '--k')
plt.scatter([y0], [y1], c='b')
plt.scatter([y2], [2.2], c='b')
plt.plot(dom, 2.2*np.ones(1000), '-.k', label=r"$\infty$")
plt.xlabel('birth')
plt.ylabel('death')
plt.legend(loc='lower right')
plt.show()
