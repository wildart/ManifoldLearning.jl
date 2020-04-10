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
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `maxoutdim`: a dimension of the reduced space.
* `t`: a number of transitions
* `α`: a normalization parameter
* `ɛ`: a Gaussian kernel variance (the scale parameter)

# Examples
```julia
M = fit(DiffMap, rand(3,100)) # construct diffusion map model
R = transform(M)              # perform dimensionality reduction
```
"""
function fit(::Type{DiffMap}, X::AbstractMatrix{T}; maxoutdim::Int=2, t::Int=1, α::Real=0.0, ɛ::Real=1.0) where {T<:Real}
    # compute kernel matrix
    sumX = sum(X.^ 2, dims=1)
    L = exp.(-( transpose(sumX) .+ sumX .- 2*transpose(X) * X ) ./ convert(T, ɛ))
    # L = pairwise((x,y)-> exp(-norm(x-y,2)^2/ε), Xtr)

    # Calculate Laplacian & normalize it
    if α > 0
        D = transpose(sum(L, dims=1))
        L ./= (D * transpose(D)) .^ convert(T, α)
    end
    D = Diagonal(vec(sum(L, dims=1)))
    M = inv(D)*L

    # D = Diagonal(vec(sum(L, dims=1)))
    # D⁻ᵅ = inv(D^α)
    # Lᵅ = D⁻ᵅ*L*D⁻ᵅ
    # Dᵅ = Diagonal(vec(sum(Lᵅ, dims=1)))
    # M = inv(Dᵅ)*Lᵅ

    # Eigendecomposition & reduction
    F = eigen(M, permute=false, scale=false)
    λ = real.(F.values)
    idx = sortperm(λ, rev=true)[2:maxoutdim+1]
    λ = λ[idx]
    V = real.(F.vectors[:,idx])
    Y = (λ.^t) .* V'

    return DiffMap{T}(t, α, ɛ, λ, L, Y)
end

"""
    transform(R::DiffMap)

Transforms the data fitted to the diffusion map model `R` into a reduced space representation.
"""
transform(R::DiffMap) = R.proj
