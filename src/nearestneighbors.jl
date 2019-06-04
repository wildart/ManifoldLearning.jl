abstract type AbstractNearestNeighbors end

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

    D = pairwise((x,y)->norm(x-y), NN.fitted, X)

    d = Array{T}(undef, k, n)
    e = Array{Int}(undef, k, n)
    idxs = self ? (1:k) : (2:k+1)

    @inbounds for j = 1 : n
        e[:, j] = sortperm(D[:,j])[idxs]
        d[:, j] = D[e[:, j],j]
    end

    return (d, e)
end
