# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

"""

    DiffMap{T <: Real} <: AbstractDimensionalityReduction

The `DiffMap` type represents diffusion maps model constructed for `T` type data.
"""
struct DiffMap{T <: Real} <: NonlinearDimensionalityReduction
    d::Number
    t::Int
    α::Real
    ɛ::Real
    λ::AbstractVector{T}
    K::AbstractMatrix{T}
    proj::Projection{T}
end

## properties
size(R::DiffMap) = (R.d, size(R.proj, 1))
eigvals(R::DiffMap) = R.λ

## custom
"""Returns the kernel matrix of the diffusion maps model `R`"""
kernel(R::DiffMap) = R.K

## show
function summary(io::IO, R::DiffMap)
    id, od = size(R)
    print(io, "Diffusion Maps(indim = $id, outdim = $od, t = $(R.t), α = $(R.α), ɛ = $(R.ɛ))")
end
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
    fit(DiffMap, data; maxoutdim=2, t=1, α=0.0, ɛ=1.0)

Fit a isometric mapping model to `data`.

# Arguments
* `data::Matrix`: a ``d \\times n``matrix of observations. Each column of `data` is
an observation, `d` is a number of features, `n` is a number of observations.

# Keyword arguments
* `kernel::Union{Nothing, Function}`: the kernel function.
It maps two input vectors (observations) to a scalar (a metric of their similarity).
by default, a Gaussian kernel. If `kernel` set to `nothing`, we assume `data` is
instead the ``n \\times n``  precomputed Gram matrix.
* `ɛ::Real=1.0`: the Gaussian kernel variance (the scale parameter). It's ignored if the custom `kernel` is passed.
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
K = ManifoldLearning.pairwise(kernel, eachcol(X), symmetric=true)
M = fit(DiffMap, K, kernel=nothing)
```
"""
function fit(::Type{DiffMap}, X::AbstractMatrix{T};
             ɛ::Real=1.0,
             kernel::Union{Nothing, Function}=(x, y) -> exp(-sum((x .- y) .^ 2) / convert(T, ɛ)), 
             maxoutdim::Int=2, 
             t::Int=1, 
             α::Real=0.0) where {T<:Real}
    if isa(kernel, Function)
        # compute Gram matrix
        L = pairwise(kernel, eachcol(X), symmetric=true)
        d = size(X,1)
    else
        # X is the pre-computed Gram matrix
        L = deepcopy(X) # deep copy needed b/c of procedure for α > 0
        d = NaN
        @assert issymmetric(L)
    end

    # Calculate Laplacian & normalize it
    if α > 0
        normalize!(L, α=α, norm=:sym)     # Lᵅ = D⁻ᵅ*L*D⁻ᵅ
        normalize!(L, α=α, norm=:rw)      # M = inv(Dᵅ)*Lᵅ
    end
    # Eigendecomposition & reduction
    λ, V = decompose(L, maxoutdim; rev=true, skipfirst=false)
    Y = (λ .^ t) .* V'

    return DiffMap{T}(d, t, α, ɛ, λ, L, Y)
end

"""
    predict(R::DiffMap)

Transforms the data fitted to the diffusion map model `R` into a reduced space representation.
"""
predict(R::DiffMap) = R.proj
