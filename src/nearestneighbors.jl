"""
    AbstractNearestNeighbors

Abstract type for nearest neighbor plug-in implementations.
"""
abstract type AbstractNearestNeighbors end

"""
    knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Int; kwargs...) -> (D,E)

Perform construction of a distance matrix `D` and a weighted graph `E` of `k` nearest neighbors from the `NN` object.
"""
function knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Int; kwargs...) where T<:Real end

"""
    BruteForce

Calculate NN using pairwise distance matrix.
"""
struct BruteForce{T<:Real} <: AbstractNearestNeighbors
    fitted::AbstractMatrix{T}
end
show(io::IO, NN::BruteForce) = print(io, "BruteForce")
function fit(::Type{BruteForce}, X::AbstractMatrix{T}) where {T<:Real}
    BruteForce(X)
end

function knn(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Int; self=false) where T<:Real
    m = size(X,1)
    n = size(X,2)
    @assert n > k "Number of observations must be more then $(k)"

    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))

    d = Array{T}(undef, k, n)
    e = Array{Int}(undef, k, n)
    idxs = self ? (1:k) : (2:k+1)

    @inbounds for j = 1 : n
        e[:, j] = sortperm(D[:,j])[idxs]
        d[:, j] = D[e[:, j],j]
    end

    return (d, e)
end
