# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

"""

    DiffMap{T <: Real} <: AbstractDimensionalityReduction

The `DiffMap` type represents diffusion maps model constructed for `T` type data.
"""
struct DiffMap{T <: Real} <: AbstractDimensionalityReduction
    t::Int
    α::Real
    ɛ::Real
    λ::AbstractVector{T}
    K::AbstractMatrix{T}
    proj::Projection{T}
end

## properties
outdim(R::DiffMap) = size(R.proj, 1)
eigvals(R::DiffMap) = R.λ

## custom
"""Returns the kernel matrix of the diffusion maps model `R`"""
kernel(R::DiffMap) = R.K

## show
summary(io::IO, R::DiffMap) = print(io, "Diffusion Maps(outdim = $(outdim(R)), t = $(R.t), α = $(R.α), ɛ = $(R.ɛ))")
function show(io::IO, R::DiffMap)
    summary(io, R)
    io = IOContext(io, :limit=>true)
    println(io)
    println(io, "Kernel: ")
    Base.print_matrix(io, R.K, "[", ",","]")
    println(io)
    println(io, "Embedding:")
    Base.print_matrix(io, transform(R), "[", ",","]")
end

## interface functions
"""
    fit(DiffMap, data; maxoutdim=2, t=1, α=1.0, ɛ=1.0)

Fit a isometric mapping model to `data`.

# Arguments
* `data::Matrix`: a (n_features, n_observations) matrix of observations. Each column of `data` is an observation.
  if `isnothing(kernel)`, `data` is instead the (n_observations, n_observations) precomputed Gram matrix.

# Keyword arguments
* `kernel::Union{Nothing, Function}=(x, y) -> exp(-sum((x .- y) .^ 2) / ɛ)`: the kernel function. 
 maps two input vectors (observations) to a scalar (a metric of their similarity).
 by default, a Gaussian kernel. if `isnothing(kernel)`, we assume `data` is instead 
 the (n_observations, n_observations) precomputed Gram matrix.
* `ɛ::Real=1.0`: the Gaussian kernel variance (the scale parameter). ignored if custom `kernel` passed.
* `maxoutdim::Int=2`: the dimension of the reduced space.
* `t::Int=1`: the number of transitions
* `α::Real=0.0`: a normalization parameter

# Examples
```julia
X = rand(3, 100)     # toy data matrix, 100 observations

# default kernel
M = fit(DiffMap, X)  # construct diffusion map model
R = transform(M)     # perform dimensionality reduction

# custom kernel
kernel = (x, y) -> x' * y # linear kernel
M = fit(DiffMap, X, kernel=kernel)

# precomputed Gram matrix
kernel = (x, y) -> x' * y # linear kernel
K = StatsBase.pairwise(kernel, eachcol(X), symmetric=true) # custom Gram matrix
M = fit(DiffMap, K, kernel=nothing)
```
"""
function fit(::Type{DiffMap}, X::AbstractMatrix{T};
             ɛ::Real=1.0,
             kernel::Union{Nothing, Function}=(x, y) -> exp(-sum((x .- y) .^ 2) / ɛ), 
             maxoutdim::Int=2, 
             t::Int=1, 
             α::Real=0.0
            ) where {T<:Real}
    if isa(kernel, Function)
        # compute Gram matrix
        L = StatsBase.pairwise(kernel, eachcol(X), symmetric=true)
    else
        # X is the pre-computed Gram matrix
        L = deepcopy(X) # deep copy needed b/c of procedure for α > 0
        @assert issymmetric(L)
    end

    # Calculate Laplacian & normalize it
    if α > 0
        D = transpose(sum(L, dims=1))
        L ./= (D * transpose(D)) .^ convert(T, α)
    end
    D = Diagonal(vec(sum(L, dims=1)))
    M = inv(D) * L # normalize rows to interpret as transition probabilities

    # D = Diagonal(vec(sum(L, dims=1)))
    # D⁻ᵅ = inv(D^α)
    # Lᵅ = D⁻ᵅ*L*D⁻ᵅ
    # Dᵅ = Diagonal(vec(sum(Lᵅ, dims=1)))
    # M = inv(Dᵅ)*Lᵅ

    # Eigendecomposition & reduction
    F = eigen(M, permute=false, scale=false)
    # for symmetric matrix, eigenvalues should be real but owing to numerical imprecision, could have nonzero-imaginary parts.
    λ = real.(F.values) 
    idx = sortperm(λ, rev=true)[2:maxoutdim+1]
    λ = λ[idx]
    V = real.(F.vectors[:, idx])
    Y = (λ .^ t) .* V'

    return DiffMap{T}(t, α, ɛ, λ, L, Y)
end

"""
    transform(R::DiffMap)

Transforms the data fitted to the diffusion map model `R` into a reduced space representation.
"""
transform(R::DiffMap) = R.proj
