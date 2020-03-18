using Plots

import FLANN
function knn_flann(X::AbstractMatrix{T}, k::Int=12) where T<:Real
    params = FLANN.FLANNParameters()
    E, D = FLANN.knn(X, X, k+1, params)
    sqrt.(@view D[2:end, :]), @view E[2:end, :]
end

import NearestNeighbors
function knn_nearestneighbors(X::AbstractMatrix{T}, k::Int=12) where T<:Real
    n = size(X,2)
    kdtree = NearestNeighbors.KDTree(X)
    idxs, dist = NearestNeighbors.knn(kdtree, X, k+1, true)
    D = Array{T}(undef, k, n)
    E = Array{Int32}(undef, k, n)
    for i in eachindex(idxs)
        E[:, i] = idxs[i][2:end]
        D[:, i] = dist[i][2:end]
    end
    return D, E
end

using ManifoldLearning
k=13
X, L = ManifoldLearning.swiss_roll()

# Use default distance matrix based method to find nearest neighbors
M1 = fit(Isomap, X)
Y1 = transform(M1)

# Use NearestNeighbors package to find nearest neighbors
M2 = fit(Isomap, X, knn=knn_nearestneighbors)
Y2 = transform(M2)

# Use FLANN package to find nearest neighbors
M3 = fit(Isomap, X, knn=knn_flann)
Y3 = transform(M3)

plot(
    plot(X[1,:], X[2,:], X[3,:], zcolor=L, m=2, t=:scatter3d, leg=false, title="Swiss Roll"),
    plot(Y1[1,:], Y1[2,:], c=L, m=2, t=:scatter, title="Distance Matrix"),
    plot(Y2[1,:], Y2[2,:], c=L, m=2, t=:scatter, title="NearestNeighbors"),
    plot(Y3[1,:], Y3[2,:], c=L, m=2, t=:scatter, title="FLANN")
, leg=false)
