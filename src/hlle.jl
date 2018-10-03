# Hessian Eigenmaps (HLLE)
# ---------------------------
# Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data,
# D. Donoho and C. Grimes, Proc Natl Acad Sci U S A. 2003 May 13; 100(10): 5591–5596

import Combinatorics: combinations

#### HLLE type
struct HLLE{T <: AbstractFloat} <: SpectralResult
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}

    HLLE{T}(k::Int, λ::AbstractVector{T}, proj::Projection{T}) where T = new(k, λ, proj)
end

## properties
outdim(M::HLLE) = size(M.proj, 1)
projection(M::HLLE) = M.proj

eigvals(M::HLLE) = M.λ
neighbors(M::HLLE) = M.k

## show & dump
function show(io::IO, M::HLLE)
    print(io, "Hessian Eigenmaps(outdim = $(outdim(M)), neighbors = $(neighbors(M)))")
end

function dump(io::IO, M::HLLE)
    show(io, M)
    println(io, "eigenvalues: ")
    Base.showarray(io, transpose(M.λ), header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function transform(::Type{HLLE}, X::DenseMatrix{T}; d::Int=2, k::Int=12) where T<:AbstractFloat
    n = size(X, 2)

    # Identify neighbors
    D, I = find_nn(X, k)

    # Obtain tangent coordinates and develop Hessian estimator
    hs = round(Int, d*(d+1)/2)
    W = spzeros(hs*n,n)
    for i=1:n
        # re-center points in neighborhood
        μ = mean(X[:,I[:,i]], dims=2)
        N = X[:,I[:,i]] .- μ
        # calculate tangent coordinates
        #tc = svd(transpose(N)).U[:,1:d]
        tc = svd(N).V[:,1:d]

        # Develop Hessian estimator
        Yi = [ones(k) tc zeros(k,hs)]
        for ii=1:d
            Yi[:,d+ii+1] = tc[:,ii].^2
        end
        yi = 2(1+d)
        for (ii,jj) in combinations(1:d,2)
            Yi[:, yi] = tc[:, ii] .* tc[:, jj]
            yi += 1
        end
        F = qr(Yi)
        H = transpose(Matrix(F.Q)[:,d+2:end])
        W[(i-1)*hs .+ (1:hs),I[:,i]] = H
    end

    # decomposition
    λ, V = decompose(transpose(W)*W, d)
    return HLLE{T}(k, λ, transpose(V) .* sqrt(n))
end
