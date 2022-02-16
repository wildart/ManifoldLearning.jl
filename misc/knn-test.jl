using ManifoldLearning

include("nearestneighbors.jl")

X, L = ManifoldLearning.swiss_roll(;segments=5)

# Use default distance matrix based method to find nearest neighbors
Y1 = predict(fit(Isomap, X, k=10))

# Use NearestNeighbors package to find nearest neighbors
Y2 = predict(fit(Isomap, X, nntype=KDTree, k=10))

# Use FLANN package to find nearest neighbors
Y3 = predict(fit(Isomap, X, nntype=FLANNTree, k=8))

using Plots
plot(
    scatter3d(X[1,:], X[2,:], X[3,:], zcolor=L, m=2, leg=:none, camera=(10,10), title="Swiss Roll"),
    scatter(Y1[1,:], Y1[2,:], c=L, m=2, title="Distance Matrix"),
    scatter(Y2[1,:], Y2[2,:], c=L, m=2, title="NearestNeighbors"),
    scatter(Y3[1,:], Y3[2,:], c=L, m=2, title="FLANN")
, leg=false)
