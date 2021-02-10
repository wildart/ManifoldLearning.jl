using ManifoldLearning

include("nearestneighbors.jl")

k=13
X, L = ManifoldLearning.swiss_roll()

# Use default distance matrix based method to find nearest neighbors
M1 = fit(Isomap, X)
Y1 = transform(M1)

# Use NearestNeighbors package to find nearest neighbors
M2 = fit(Isomap, X, nntype=KDTree)
Y2 = transform(M2)

# Use FLANN package to find nearest neighbors
M3 = fit(Isomap, X, nntype=FLANNTree)
Y3 = transform(M3)

using Plots
plot(
    plot(X[1,:], X[2,:], X[3,:], zcolor=L, m=2, t=:scatter3d, leg=false, title="Swiss Roll"),
    plot(Y1[1,:], Y1[2,:], c=L, m=2, t=:scatter, title="Distance Matrix"),
    plot(Y2[1,:], Y2[2,:], c=L, m=2, t=:scatter, title="NearestNeighbors"),
    plot(Y3[1,:], Y3[2,:], c=L, m=2, t=:scatter, title="FLANN")
, leg=false)
