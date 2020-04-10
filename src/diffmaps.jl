# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

"""

    DiffMap{T <: Real} <: AbstractDimensionalityReduction

The `DiffMap` type represents diffusion maps model with `T` type data.
"""
struct DiffMap{T <: Real} <: AbstractDimensionalityReduction
    t::Int
    ɛ::T
    K::AbstractMatrix{T}
    proj::Projection{T}
end

## properties
outdim(R::DiffMap) = size(R.proj, 1)

## custom
"""Returns the kernel matrix of the diffusion maps model `R`"""
kernel(R::DiffMap) = R.K

## show
summary(io::IO, R::DiffMap) = print(io, "Diffusion Maps(outdim = $(outdim(R)), t = $(R.t), ɛ = $(R.ɛ))")
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
    fit(DiffMap, data; maxoutdim=2, t=1, ɛ=1.0)

Fit a isometric mapping model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `maxoutdim`: a dimension of the reduced space.
* `t`: a number of transitions
* `ɛ`: a Gaussian kernel variance (the scale parameter)

# Examples
```julia
M = fit(DiffMap, rand(3,100)) # construct diffusion map model
R = transform(M)              # perform dimensionality reduction
```
"""
function fit(::Type{DiffMap}, X::AbstractMatrix{T}; maxoutdim::Int=2, t::Int=1, ɛ::Real=1.0) where {T<:Real}
    # rescale data
    Xtr = standardize(StatsBase.UnitRangeTransform, X; dims=2)
    Xtr[findall(isnan, Xtr)] .= 0

    # compute kernel matrix
    sumX = sum(Xtr.^ 2, dims=1)
    K = exp.(-( transpose(sumX) .+ sumX .- 2*transpose(Xtr) * Xtr ) ./ convert(T, ɛ))

    p = transpose(sum(K, dims=1))
    K ./= (p * transpose(p)) .^ convert(T, t)
    p = transpose(sqrt.(sum(K, dims=1)))
    K ./= p * transpose(p)

    U, S, V = svd(K, full=true)
    U ./= U[:,1]
    Y = U[:,2:(maxoutdim+1)]

    return DiffMap{T}(t, convert(T, ɛ), K, transpose(Y))
end

"""
    transform(R::DiffMap)

Transforms the data fitted to the diffusion map model `R` into a reduced space representation.
"""
transform(R::DiffMap) = R.proj
