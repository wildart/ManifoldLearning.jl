# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

#### DiffMap type
struct DiffMap{T <: AbstractFloat} <: SpectralResult
    t::Int
    ɛ::Float64
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
function transform{T<:AbstractFloat}(::Type{DiffMap}, X::DenseMatrix{T};
                                     d::Int=2, t::Int=1, ɛ::T=1.0)
    transform!(fit(UnitRangeTransform, X), X)

    sumX = sum(X.^ 2, 1)
    K = exp.(( sumX' .+ sumX .- 2*At_mul_B(X,X) ) ./ ɛ)

    p = sum(K, 1)'
    K ./= ((p * p') .^ t)
    p = sqrt.(sum(K, 1))'
    K ./= (p * p')

    U, S, V = svd(K, thin=false)
    U ./= U[:,1]
    Y = U[:,2:(d+1)]

    return DiffMap{T}(t, ɛ, K, Y')
end
