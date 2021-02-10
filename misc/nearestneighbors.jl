using ManifoldLearning
import StatsBase

import NearestNeighbors
struct KDTree <: ManifoldLearning.AbstractNearestNeighbors
    k::Integer
    fitted::NearestNeighbors.KDTree
end
Base.show(io::IO, NN::KDTree) = print(io, "KDTree(k=$(NN.k))")
StatsBase.fit(::Type{KDTree}, X::AbstractMatrix{T}, k::Integer) where {T<:Real} = KDTree(k, NearestNeighbors.KDTree(X))
function ManifoldLearning.knn(NN::KDTree, X::AbstractVecOrMat{T}; self=false) where {T<:Real}
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

import FLANN
struct FLANNTree{T <: Real} <: ManifoldLearning.AbstractNearestNeighbors
    k::Integer
    index::FLANN.FLANNIndex{T}
end
Base.show(io::IO, NN::FLANNTree) = print(io, "FLANNTree(k=$(NN.k))")
function StatsBase.fit(::Type{FLANNTree}, X::AbstractMatrix{T}, k::Integer) where {T<:Real}
    params = FLANNParameters()
    idx = FLANN.flann(X, params)
    FLANNTree(k, idx)
end
function ManifoldLearning.knn(NN::FLANNTree, X::AbstractVecOrMat{T}; self=false) where {T<:Real}
    E, D = FLANN.knn(NN.index, X, NN.k+1)
    sqrt.(@view D[2:end, :]), @view E[2:end, :]
end
