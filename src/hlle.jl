# Hessian Eigenmaps (HLLE)
# ---------------------------
# Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data,
# D. Donoho and C. Grimes, Proc Natl Acad Sci U S A. 2003 May 13; 100(10): 5591–5596

import Combinatorics: combinations

#### HLLE type
struct HLLE{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}

    HLLE{T}(k::Int, λ::AbstractVector{T}, proj::Projection{T}) where T = new(k, λ, proj)
end

## properties
outdim(R::HLLE) = size(R.proj, 1)
eigvals(R::HLLE) = R.λ
neighbors(R::HLLE) = R.k

## show
summary(io::IO, R::HLLE) = print(io, "Hessian Eigenmaps(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
function fit(::Type{HLLE}, X::AbstractMatrix{T}; maxoutdim::Int=2, k::Int=12, knn=knn) where {T<:Real}
    n = size(X, 2)

    # Identify neighbors
    D, E = knn(X, k)

    # Obtain tangent coordinates and develop Hessian estimator
    hs = (maxoutdim*(maxoutdim+1)) >> 1
    W = spzeros(T, hs*n, n)
    for i=1:n
        II = @view E[:,i]
        # re-center points in neighborhood
        μ = mean(X[:, II], dims=2)
        N = X[:, II] .- μ
        # calculate tangent coordinates
        tc = svd(N).V[:,1:maxoutdim]

        # Develop Hessian estimator
        Yi = [ones(T, k) tc zeros(T, k, hs)]
        for ii=1:maxoutdim
            Yi[:,maxoutdim+ii+1] = tc[:,ii].^2
        end
        yi = 2*(1+maxoutdim)
        for (ii,jj) in combinations(1:maxoutdim, 2)
            Yi[:, yi] = tc[:, ii] .* tc[:, jj]
            yi += 1
        end
        F = qr(Yi)
        H = transpose(F.Q[:,(end-(hs-1)):end])
        W[(1:hs).+(i-1)*hs, II] = H
    end

    # decomposition
    λ, V = decompose(transpose(W)*W, maxoutdim)
    return HLLE{T}(k, λ, transpose(V) .* convert(T, sqrt(n)))
end

transform(R::HLLE) = R.proj
