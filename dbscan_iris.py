import time
import numpy as np
from dbscan import DBScan
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import cycle, islice

np.random.seed(0)

iris = datasets.load_iris()
X = iris.data[:, :2]  # Looking at only Sepal Length and Width for now
y = iris.target

plt.figure()
t0 = time.time()
scanner = DBScan(.2, 4)

labels = scanner.fit(X)
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(labels) + 1))))
# add black color for outliers (if any)
colors = np.append(colors, ["#000000"])
plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])
t1 = time.time()
plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                transform=plt.gca().transAxes, size=15,
                horizontalalignment='right')
plt.show()