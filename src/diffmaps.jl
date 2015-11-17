# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

#### DiffMap type
immutable DiffMap{T <: Real} <: SpectralResult
    t::Int
    ɛ::Float64
    K::AbstractMatrix{T}
    proj::Projection{T}

    DiffMap{T}(t::Int, ɛ::T, K::AbstractMatrix{T}, proj::Projection{T}) = new(t, ɛ, K, proj)
end

## properties
outdim(M::DiffMap) = size(M.proj, 1)
projection(M::DiffMap) = M.proj
kernel(M::DiffMap) = M.K

## show & dump
function show(io::IO, M::DiffMap)
    print(io, "Diffusion Maps(outdim = $(outdim(M)), t = $(M.t), ɛ = $(M.ɛ))")
end

function dump(io::IO, M::DiffMap)
    show(io, M)
    println(io, "kernel: ")
    Base.showarray(io, M.K, header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, projection(M), header=false, repr=false)
end

## interface functions
function transform{T<:Real}(::Type{DiffMap}, X::DenseMatrix{T}; d::Int=2, t::Int=1, ɛ::T=1.0)
    transform!(fit(UnitRangeTransform, X), X)

    sumX = sum(X.^ 2, 1)
    K = exp(( sumX' .+ sumX .- 2*At_mul_B(X,X) ) ./ ɛ)

    p = sum(K, 1)'
    K ./= ((p * p') .^ t)
    p = sqrt(sum(K, 1))'
    K ./= (p * p')

    U, S, V = svd(K, thin=false)
    U ./= U[:,1]
    Y = U[:,2:(d+1)]

    return DiffMap{T}(t, ɛ, K, Y')
end
