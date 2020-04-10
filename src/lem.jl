# Laplacian Eigenmaps
# -------------------
# Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
# M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396

"""
    LEM{NN <: AbstractNearestNeighbors, T <: Real} <: AbstractDimensionalityReduction

The `LEM` type represents a Laplacian eigenmaps model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct LEM{NN <: AbstractNearestNeighbors, T <: Real} <: AbstractDimensionalityReduction
    λ::AbstractVector{T}
    ɛ::T
    proj::Projection{T}
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
outdim(R::LEM) = size(R.proj, 1)
eigvals(R::LEM) = R.λ
neighbors(R::LEM) = R.nearestneighbors.k
vertices(R::LEM) = R.component

## show
summary(io::IO, R::LEM) = print(io, "Laplacian Eigenmaps(outdim = $(outdim(R)), neighbors = $(neighbors(R)), ɛ = $(R.ɛ))")

## interface functions
"""
    fit(LEM, data; k=12, maxoutdim=2, ɛ=1.0, nntype=BruteForce)

Fit a Laplacian eigenmaps model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)
* `ɛ`: a Gaussian kernel variance (the scale parameter)

# Examples
```julia
M = fit(LEM, rand(3,100)) # construct Laplacian eigenmaps model
R = transform(M)          # perform dimensionality reduction
```
"""
function fit(::Type{LEM}, X::AbstractMatrix{T};
        k::Int=12, maxoutdim::Int=2, ɛ::Real=1.0, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    NN = fit(nntype, X, k)
    D, E = knn(NN, X)
    G, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))

    # Compute weights
    W = weights(G)
    W .^= 2
    W ./= maximum(W)

    W[W .> eps(T)] = exp.(-W[W .> eps(T)] ./ convert(T,ɛ))
    D = spdiagm(0=>vec(sum(W, dims=2)))
    L = D - W

    λ, V = decompose(L, D, maxoutdim)
    return LEM{nntype, T}(λ, ɛ, transpose(V), NN, C)
end

"""
    transform(R::LEM)

Transforms the data fitted to the Laplacian eigenmaps model `R` into a reduced space representation.
"""
transform(R::LEM) = R.proj
