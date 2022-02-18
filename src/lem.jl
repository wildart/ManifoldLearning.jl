# Laplacian Eigenmaps
# -------------------
# Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
# M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396

"""
    LEM{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction

The `LEM` type represents a Laplacian eigenmaps model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct LEM{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction
    d::Int
    k::Real
    λ::AbstractVector{T}
    ɛ::T
    proj::Projection{T}
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
size(R::LEM) = (R.d, size(R.proj, 1))
eigvals(R::LEM) = R.λ
neighbors(R::LEM) = R.k
vertices(R::LEM) = R.component

## show
function summary(io::IO, R::LEM)
    id, od = size(R)
    msg = isinteger(R.k) ? "neighbors" : "epsilon"
    print(io, "LEM{$(R.nearestneighbors)}(indim = $id, outdim = $od, $msg = $(R.k))")
end


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
* `laplacian`: a form of the Laplacian matrix used for spectral decomposition
   * `:unnorm`: an unnormalized Laplacian
   * `:sym`: a symmetrically normalized Laplacian
   * `:rw`: a random walk normalized Laplacian

# Examples
```julia
M = fit(LEM, rand(3,100)) # construct Laplacian eigenmaps model
R = predict(M)          # perform dimensionality reduction
```
"""
function fit(::Type{LEM}, X::AbstractMatrix{T}; k::Real=12, maxoutdim::Int=2,
             ɛ::Real=1, laplacian::Symbol=:unnorm, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    d, n = size(X)
    NN = fit(nntype, X)
    A = adjacency_matrix(NN, X, k)
    G, C = largest_component(SimpleGraph(A))

    # Compute weights of heat kernel
    W = A[C,C]
    I, J, V = findnz(W)
    @inbounds for (i,j,v) in zip(I,J,V)
        W[i,j] = exp(-v*v/ε)
    end

    L, D = Laplacian(W)
    λ, V = if laplacian == :unnorm
        decompose(L, collect(D), maxoutdim)
    elseif laplacian == :sym
        normalize!(L, D, α=1/2, norm=laplacian)
        decompose(L, maxoutdim)
    elseif laplacian == :rw
        normalize!(L, D, α=1, norm=laplacian)
        decompose(L, maxoutdim)
    else
        throw(ArgumentError("Unkown Laplacian type: $laplacian"))
    end
    return LEM{nntype, T}(d, k, λ, ɛ, transpose(V), NN, C)
end

"""
    predict(R::LEM)

Transforms the data fitted to the Laplacian eigenmaps model `R` into a reduced space representation.
"""
predict(R::LEM) = R.proj

