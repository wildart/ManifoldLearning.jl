# Additional wrappers for performant calculations of nearest neighbors 

using ManifoldLearning
import Base: show
import StatsAPI: fit
import ManifoldLearning: knn

# Wrapper around NearestNeighbors functionality
using NearestNeighbors: NearestNeighbors
struct KDTree <: ManifoldLearning.AbstractNearestNeighbors
    fitted::NearestNeighbors.KDTree
end
show(io::IO, NN::KDTree) = print(io, "KDTree")
fit(::Type{KDTree}, X::AbstractMatrix{T}) where {T<:Real} = KDTree(NearestNeighbors.KDTree(X))
function knn(NN::KDTree, X::AbstractVecOrMat{T}, k::Int; self=false) where {T<:Real}
    m, n = size(X)
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
    index::FLANN.FLANNIndex{T}
end
show(io::IO, NN::FLANNTree) = print(io, "FLANNTree")
function fit(::Type{FLANNTree}, X::AbstractMatrix{T}) where {T<:Real}
    params = FLANN.FLANNParameters()
    idx = FLANN.flann(X, params)
    FLANNTree(idx)
end
function knn(NN::FLANNTree, X::AbstractVecOrMat{T}, k::Int; self=false) where {T<:Real}
    E, D = FLANN.knn(NN.index, X, NN.k+1)
    sqrt.(@view D[2:end, :]), @view E[2:end, :]
end

