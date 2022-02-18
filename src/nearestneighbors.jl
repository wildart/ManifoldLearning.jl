"""
    AbstractNearestNeighbors

Abstract type for nearest neighbor plug-in implementations.
"""
abstract type AbstractNearestNeighbors end

"""
    adjacency_matrix(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Real; kwargs...) -> A

Perform construction of a weighted adjacency distance matrix `A` from the points
in `X` given the `NN` object.
- If `k` is a positive integer, then `k` nearest neighbors are use for construction.
- If `k` is a real number, then radius `k` neighborhood is used for construction.

"""
function knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer; kwargs...) where T<:Real end

"""
    adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Real; kwargs...) -> (A, W)

Perform construction of an adjacency list `A` with corresponding weights `W` from
the points in `X` given the `NN` object.
- If `k` is a positive integer, then `k` nearest neighbors are use for construction.
- If `k` is a real number, then radius `k` neighborhood is used for construction.

"""
function adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer; kwargs...) where T<:Real end

"""
    BruteForce

Calculate nearest neighborhoods using pairwise distance matrix.
"""
struct BruteForce{T<:Real} <: AbstractNearestNeighbors
    fitted::AbstractMatrix{T}
end
show(io::IO, NN::BruteForce) = print(io, "BruteForce")
function fit(::Type{BruteForce}, X::AbstractMatrix{T}) where {T<:Real}
    BruteForce(X)
end

function adjacency_list(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Integer;
                        self=false) where T<:Real
    n = size(NN.fitted,2)
    @assert n > k "Number of observations must be more then $(k)"

    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))
    idxs = self ? (1:k) : (2:k+1)

    A = Vector{Vector{Int}}(undef, n)
    W = Vector{Vector{T}}(undef, n)
    @inbounds for (j, ds) in pairs(eachcol(D))
        kidxs = sortperm(ds)[idxs]
        A[j] = kidxs
        W[j] = D[kidxs, j]
    end
    return A, W
end

function adjacency_list(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Real;
                        self=false) where T<:Real
    n = size(NN.fitted,2)
    @assert n > k "Number of observations must be more then $(k)"

    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))
    idxs = self ? (1:k) : (2:k+1)

    A = Vector{Vector{Int}}(undef, n)
    W = Vector{Vector{T}}(undef, n)
    @inbounds for (j, ds) in pairs(eachcol(D))
        kidxs = self ? findall(0 .<= ds .<= k) : findall(0 .< ds .<= k)
        A[j] = kidxs
        W[j] = D[kidxs, j]
    end
    return A, W
end

function adjacency_matrix(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Integer;
                          self=false) where T<:Real
    n = size(NN.fitted,2)
    @assert n > k "Number of observations must be more then $(k)"

    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))
    idxs = self ? (1:k) : (2:k+1)

    A = spzeros(T, n, n)
    @inbounds for (j, ds) in pairs(eachcol(D))
        kidxs = sortperm(ds)[idxs]
        w = @view D[kidxs, j]
        A[kidxs, j] .= w
        A[j, kidxs] .= w
    end
    return A
end

function adjacency_matrix(NN::BruteForce{T}, X::AbstractVecOrMat{T}, k::Real;
                          self=false) where T<:Real
    n = size(X,2)
    D = pairwise((x,y)->norm(x-y), eachcol(NN.fitted), eachcol(X))

    A = spzeros(T, n, n)
    @inbounds for (j, ds) in pairs(eachcol(D))
        idxs = self ? findall(0 .<= ds .<= k) : findall(0 .< ds .<= k)
        w = @view D[idxs, j]
        A[idxs, j] .= w
        A[j, idxs] .= w
    end
    return A
end

