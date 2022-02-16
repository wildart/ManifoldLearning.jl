# Additional wrappers for performant calculations of nearest neighbors 

using ManifoldLearning
import Base: show
import StatsAPI: fit
import ManifoldLearning: knn

# Wrapper around NearestNeighbors functionality
using NearestNeighbors: NearestNeighbors
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

# Wrapper around FLANN functionality
using FLANN: FLANN
struct FLANNTree{T <: Real} <: ManifoldLearning.AbstractNearestNeighbors
    k::Integer
    index::FLANN.FLANNIndex{T}
end
show(io::IO, NN::FLANNTree) = print(io, "FLANNTree(k=$(NN.k))")
function fit(::Type{FLANNTree}, X::AbstractMatrix{T}, k::Integer) where {T<:Real}
    params = FLANN.FLANNParameters()
    idx = FLANN.flann(X, params)
    FLANNTree(k, idx)
end
function knn(NN::FLANNTree, X::AbstractVecOrMat{T}; self=false) where {T<:Real}
    E, D = FLANN.knn(NN.index, X, NN.k+1)
    sqrt.(@view D[2:end, :]), @view E[2:end, :]
end

