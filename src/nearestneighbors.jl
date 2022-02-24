"""
    AbstractNearestNeighbors

Abstract type for nearest neighbor plug-in implementations.
"""
abstract type AbstractNearestNeighbors end

"""
    size(NN::AbstractNearestNeighbors)

Returns the size of the fitted data.
"""
function size(NN::AbstractNearestNeighbors) end

"""
    knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer; kwargs...) -> (I,D)

Returns `(k, n)`-matrices of point indexes and distances of `k` nearest neighbors
for points in the `(m,n)`-matrix `X` given the `NN` object.
"""
function knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer; kwargs...) where T<:Real end

"""
    inrange(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, r::Real; kwargs...) -> (I,D)

Returns collections of point indexes and distances in radius `r` of points in
the `(m,n)`-matrix `X` given the `NN` object.
"""
function inrange(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, r::Real; kwargs...) where T<:Real end

"""
    adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Real; kwargs...) -> (A, W)

Perform construction of an adjacency list `A` with corresponding weights `W` from
the points in `X` given the `NN` object.
- If `k` is a positive integer, then `k` nearest neighbors are use for construction.
- If `k` is a real number, then radius `k` neighborhood is used for construction.

"""
function adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T},
                        k::Integer; weights::Bool=false, kwargs...) where T<:Real
    A, W = knn(NN, X, k; weights=weights, kwargs...)
    return A, W
end
function adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T},
                        k::Real; weights::Bool=false, kwargs...) where T<:Real
    A, W = inrange(NN, X, k; weights=weights, kwargs...)
    return A, W
end

"""
    adjacency_matrix(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Real; kwargs...) -> A

Perform construction of a weighted adjacency distance matrix `A` from the points
in `X` given the `NN` object.
- If `k` is a positive integer, then `k` nearest neighbors are use for construction.
- If `k` is a real number, then radius `k` neighborhood is used for construction.

"""
function adjacency_matrix(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T},
                          k::Integer; symmetric::Bool=true, kwargs...) where T<:Real
    n = size(NN)[2]
    m = length(eachcol(X))
    @assert n >=m "Cannot construc matrix for more then $n fitted points"
    E, W = knn(NN, X, k; weights=true, kwargs...)
    return sparse(E, W, n, symmetric=symmetric)
end
function adjacency_matrix(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T},
                          r::Real; symmetric::Bool=true, kwargs...) where T<:Real
    n = size(NN)[2]
    m = length(eachcol(X))
    @assert n >=m "Cannot construc matrix for more then $n fitted points"
    E, W = inrange(NN, X, r; weights=true, kwargs...)
    return sparse(E, W, n, symmetric=symmetric)
end


# Implementation
"""
    BruteForce

Calculate nearest neighborhoods using pairwise distance matrix.
"""
struct BruteForce{T<:Real} <: AbstractNearestNeighbors
    fitted::AbstractMatrix{T}
end
show(io::IO, NN::BruteForce) = print(io, "BruteForce")
size(NN::BruteForce) = size(NN.fitted)
fit(::Type{BruteForce}, X::AbstractMatrix{T}) where {T<:Real} = BruteForce(X)

function knn(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Integer;
             self::Bool=false, weights::Bool=true, kwargs...) where T<:Real
    l = size(NN)[2]
    @assert l > k "Number of fitted observations must be at least $(k)"

    # construct distance matrix
    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))
    idxs = (1:k).+(!self)

    n = size(X,2)
    A = Vector{Vector{Int}}(undef, n)
    W = Vector{Vector{T}}(undef, (weights ? n : 0))
    @inbounds for (j, ds) in enumerate(eachcol(D))
        kidxs = sortperm(ds)[idxs]
        A[j] = kidxs
        if weights
            W[j] = D[kidxs, j]
        end
    end

    return A, W
end

function inrange(NN::BruteForce{T}, X::AbstractVecOrMat{T}, r::Real;
                  self::Bool=false, weights::Bool=false, kwargs...) where T<:Real
    # construct distance matrix
    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))

    n = size(X,2)
    A = Vector{Vector{Int}}(undef, n)
    W = Vector{Vector{T}}(undef, (weights ? n : 0))
    @inbounds for (j, ds) in enumerate(eachcol(D))
        kidxs = self ? findall(0 .<= ds .<= r) : findall(0 .< ds .<= r)
        A[j] = kidxs
        if weights
            W[j] = D[kidxs, j]
        end
    end
    return A, W
end

