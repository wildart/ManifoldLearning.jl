from sklearn import manifold, datasets
import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils import check_array
from sklearn.utils.graph import graph_shortest_path

n_points = 1000
n_neighbors = 10
n_components = 2

X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
# X, _ = datasets.load_digits(return_X_y=True)
X.shape
#np.savetxt('misc/data.csv', X, delimiter=',')
X = np.genfromtxt('misc/data.csv', delimiter=',')


embedding = manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors)
Y = embedding.fit_transform(X[:20])
Y

embedding.transform(X[:20])

A = check_array(X[:20])
distances, indices = embedding.nbrs_.kneighbors(A, return_distance=True)

G_X = np.zeros((A.shape[0], embedding.training_data_.shape[0]))
for i in range(A.shape[0]):
    G_X[i] = np.min(embedding.dist_matrix_[indices[i]] + distances[i][:, None], 0)

np.savetxt('misc/G.csv', G_X, delimiter=',')

G_X **= 2
G_X *= -0.5

embedding.kernel_pca_.transform(G_X)
