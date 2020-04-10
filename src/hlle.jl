# Hessian Eigenmaps (HLLE)
# ---------------------------
# Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data,
# D. Donoho and C. Grimes, Proc Natl Acad Sci U S A. 2003 May 13; 100(10): 5591–5596

import Combinatorics: combinations

"""
    HLLE{NN <: AbstractNearestNeighbors, T <: Real} <: AbstractDimensionalityReduction

The `HLLE` type represents a Hessian eigenmaps model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct HLLE{NN <: AbstractNearestNeighbors, T <: Real} <: AbstractDimensionalityReduction
    λ::AbstractVector{T}
    proj::Projection{T}
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
outdim(R::HLLE) = size(R.proj, 1)
eigvals(R::HLLE) = R.λ
neighbors(R::HLLE) = R.nearestneighbors.k
vertices(R::HLLE) = R.component

## show
summary(io::IO, R::HLLE) = print(io, "Hessian Eigenmaps(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
"""
    fit(HLLE, data; k=12, maxoutdim=2, ɛ=1.0, nntype=BruteForce)

Fit a Hessian eigenmaps model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)

# Examples
```julia
M = fit(HLLE, rand(3,100)) # construct Laplacian Eigenmaps model
R = transform(M)          # perform dimensionality reduction
```
"""
function fit(::Type{HLLE}, X::AbstractMatrix{T};
             k::Int=12, maxoutdim::Int=2, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    NN = fit(nntype, X, k)
    D, E = knn(NN, X)
    G, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))
    XX = @view X[:, C]
    n = length(C)

    # Obtain tangent coordinates and develop Hessian estimator
    hs = (maxoutdim*(maxoutdim+1)) >> 1
    W = spzeros(T, hs*n, n)
    for i=1:n
        II = @view E[:,C[i]]
        # re-center points in neighborhood
        VX = view(XX, :, II)
        μ = mean(VX, dims=2)
        N = VX .- μ
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
    return HLLE{nntype, T}(λ, transpose(V) .* convert(T, sqrt(n)), NN, C)
end

"""
    transform(R::LLE)

Transforms the data fitted to the Hessian eigenmaps model `R` into a reduced space representation.
"""
transform(R::HLLE) = R.proj
