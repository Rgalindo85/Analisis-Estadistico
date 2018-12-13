import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the data set/file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data_frame = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

#standarize the data
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

x = data_frame.loc[:, features].values
y = data_frame.loc[:, ['target']].values

x = StandardScaler().fit_transform(x)


#PCA projection to 2D
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, data_frame[['target']]], axis = 1)

#Plot the projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r','g','b']

for target, color, in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c = color,
               s = 50)

ax.legend(targets)
ax.grid()
#plt.show()

print('variance: ', pca.explained_variance_)
print('components: ', pca.components_)
#Covariance Matrix
cov_mat = np.cov(x.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('EigenValues: ', eig_vals)
print('EigenVecs: ', eig_vecs)
