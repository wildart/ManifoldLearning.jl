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
outdim(R::DiffMap) = size(R.proj, 1)

## custom
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

transform(R::DiffMap) = R.proj
