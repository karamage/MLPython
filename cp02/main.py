import pandas as pd
import Perceptron as pcpt
import matplotlib.pyplot as plt
import numpy as np
import plot as pt

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/iris/iris.data', header=None)
df.tail()

y = df.iloc[0:100, 4].values

y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50,0], X[:50,1], color='red', marker='o',
label='setosa')

plt.scatter(X[50:100,0], X[50:100,1], color='blue',
marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

plt.legend(loc='upper left')
plt.show()

ppn = pcpt.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,
marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

pt.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
