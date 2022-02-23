# Additional wrappers for calculations of nearest neighbors

using ManifoldLearning
using LinearAlgebra: norm
import Base: show, size
import StatsAPI: fit
import ManifoldLearning: knn, inradius

# Wrapper around NearestNeighbors functionality
using NearestNeighbors: NearestNeighbors
struct KDTree <: ManifoldLearning.AbstractNearestNeighbors
    fitted::AbstractMatrix
    tree::NearestNeighbors.KDTree
end
show(io::IO, NN::KDTree) = print(io, "KDTree")
fit(::Type{KDTree}, X::AbstractMatrix{T}) where {T<:Real} =
    KDTree(X, NearestNeighbors.KDTree(X))
size(NN::KDTree) = (length(NN.fitted.data[1]), length(NN.fitted.data))
function knn(NN::KDTree, X::AbstractVecOrMat{T}, k::Integer;
             self::Bool=false, weights::Bool=true, kwargs...) where {T<:Real}
    m, n = size(X)
    @assert n > k "Number of observations must be more then $(k)"
    A, D = NearestNeighbors.knn(NN.tree, X, k, true)
    return A, D
end
function inradius(NN::KDTree, X::AbstractVecOrMat{T}, r::Real;
                  weights::Bool=false, kwargs...) where {T<:Real}
    m, n = size(X)
    A = NearestNeighbors.inrange(NN.tree, X, r)
    W = Vector{Vector{T}}(undef, (weights ? n : 0))
    if weights
        for (i, ii) in enumerate(A)
            W[i] = T[]
            if length(ii) > 0
                for v in eachcol(NN.fitted[:, ii])
                    d = norm(X[:,i] - v)
                    push!(W[i], d)
                end
            end
        end
    end
    return A, W
end

#=
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
=#
