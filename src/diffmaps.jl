# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

#### DiffMap type
struct DiffMap{T <: Real} <: AbstractDimensionalityReduction
    t::Int
    ɛ::T
    K::AbstractMatrix{T}
    proj::Projection{T}

    DiffMap{T}(t::Int, ɛ::T, K::AbstractMatrix{T}, proj::Projection{T})  where T = new(t, ɛ, K, proj)
end

## properties
outdim(M::DiffMap) = size(M.proj, 1)
projection(M::DiffMap) = M.proj
kernel(M::DiffMap) = M.K

## show & dump
function show(io::IO, M::DiffMap)
    print(io, "Diffusion Maps(outdim = $(outdim(M)), t = $(M.t), ɛ = $(M.ɛ))")
end

function Base.dump(io::IO, M::DiffMap)
    println(io, "Dimensionality:")
    show(io, outdim(M))
    print(io, "\n\n")
    println(io, "Timesteps:")
    show(io, M.t)
    print(io, "\n\n")
    println(io, "Kernel: ")
    Base.showarray(io, M.K, false, header=false)
    println(io)
    println(io, "Embedding:")
    Base.showarray(io, projection(M), false, header=false)
end

## interface functions
function transform(::Type{DiffMap}, X::AbstractMatrix{T}; d::Int=2, t::Int=1, ɛ::Real=1.0) where {T<:Real}
    # rescale data
    Xtr = standardize(StatsBase.UnitRangeTransform, X)
    Xtr[findall(isnan, Xtr)] .= 0

    # compute kernel matrix
    sumX = sum(Xtr.^ 2, dims=1)
    K = exp.(( transpose(sumX) .+ sumX .- 2*transpose(Xtr) * Xtr ) ./ convert(T, ɛ))

    p = transpose(sum(K, dims=1))
    K ./= (p * transpose(p)) .^ convert(T, t)
    p = transpose(sqrt.(sum(K, dims=1)))
    K ./= p * transpose(p)

    U, S, V = svd(K, full=true)
    U ./= U[:,1]
    Y = U[:,2:(d+1)]

    return DiffMap{T}(t, convert(T, ɛ), K, transpose(Y))
end
