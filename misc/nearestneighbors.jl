using ManifoldLearning
using NearestNeighbors
using FLANN

import Base: show
import StatsAPI: fit
import ManifoldLearning: knn

struct KDTree <: ManifoldLearning.AbstractNearestNeighbors
    k::Integer
    fitted::NearestNeighbors.KDTree
end
show(io::IO, NN::KDTree) = print(io, "KDTree(k=$(NN.k))")
fit(::Type{KDTree}, X::AbstractMatrix{T}, k::Integer) where {T<:Real} = KDTree(k, NearestNeighbors.KDTree(X))
function knn(NN::KDTree, X::AbstractVecOrMat{T}; self=false) where {T<:Real}
    m, n = size(X)
    k = NN.k
    @assert n > k "Number of observations must be more then $(k)"

    idxs, dist = NearestNeighbors.knn(NN.fitted, X, k+1, true)
    D = Array{T}(undef, k, n)
    E = Array{Int32}(undef, k, n)
    for i in eachindex(idxs)
        E[:, i] = idxs[i][2:end]
        D[:, i] = dist[i][2:end]
    end
    return D, E
end

struct FLANNTree{T <: Real} <: ManifoldLearning.AbstractNearestNeighbors
    k::Integer
    index::FLANN.FLANNIndex{T}
end
show(io::IO, NN::FLANNTree) = print(io, "FLANNTree(k=$(NN.k))")
function fit(::Type{FLANNTree}, X::AbstractMatrix{T}, k::Integer) where {T<:Real}
    params = FLANNParameters()
    idx = FLANN.flann(X, params)
    FLANNTree(k, idx)
end
function knn(NN::FLANNTree, X::AbstractVecOrMat{T}; self=false) where {T<:Real}
    E, D = FLANN.knn(NN.index, X, NN.k+1)
    sqrt.(@view D[2:end, :]), @view E[2:end, :]
end

k=13
X, L = ManifoldLearning.swiss_roll()

# Use default distance matrix based method to find nearest neighbors
M1 = fit(Isomap, X)
Y1 = predict(M1)

# Use NearestNeighbors package to find nearest neighbors
M2 = fit(Isomap, X, knn=KDTree)
Y2 = predict(M2)

# Use FLANN package to find nearest neighbors
M3 = fit(Isomap, X, knn=FLANNTree)
Y3 = predict(M3)

using Plots
plot(
    plot(X[1,:], X[2,:], X[3,:], zcolor=L, m=2, t=:scatter3d, leg=false, title="Swiss Roll"),
    plot(Y1[1,:], Y1[2,:], c=L, m=2, t=:scatter, title="Distance Matrix"),
    plot(Y2[1,:], Y2[2,:], c=L, m=2, t=:scatter, title="NearestNeighbors"),
    plot(Y3[1,:], Y3[2,:], c=L, m=2, t=:scatter, title="FLANN")
, leg=false)

