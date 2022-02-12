"""
    AbstractNearestNeighbors

Abstract type for nearest neighbor plug-in implementations.
"""
abstract type AbstractNearestNeighbors end

"""
    knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}; kwargs...) -> (D,E)

Perform construction of the distance matrix `D` and neares neighbor weighted graph `E` from the `NN` object
"""
function knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}; kwargs...) where T<:Real end

"""
    BruteForce

Calculate NN using pairwise distance matrix.
"""
struct BruteForce{T<:Real} <: AbstractNearestNeighbors
    k::Integer
    fitted::AbstractMatrix{T}
end
fit(::Type{BruteForce}, X::AbstractMatrix{T}, k::Integer) where {T<:Real} = BruteForce(k, X)
show(io::IO, NN::BruteForce) = print(io, "BruteForce(k=$(NN.k))")

function knn(NN::BruteForce{T}, X::AbstractVecOrMat{T}; self=false) where T<:Real
    m, n = size(X)
    k = NN.k
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
