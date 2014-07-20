# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

#### DiffMap type
immutable DiffMap <: SpectralResult
    t::Int
    σ::Float64
    K::Matrix{Float64}
    proj::Projection

    DiffMap(t::Int, σ::Float64, K::Matrix{Float64}, proj::Projection) = new(k, σ, K, proj)
end

## properties
indim(M::DiffMap) = size(M.proj, 1)
outdim(M::DiffMap) = size(M.proj, 2)
projection(M::DiffMap) = M.proj

## show & dump
function show(io::IO, M::DiffMap)
    print(io, "Diffusion Maps(indim = $(indim(M)), outdim = $(outdim(M)), t = $(M.t), σ = $(M.σ))")
end

function dump(io::IO, M::DiffMap)
    show(io, M)
    println(io, "kernel: ")
    Base.showarray(io, M.K, header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function fit(::Type{DiffMap}, X::DenseMatrix{Float64}; d::Int=2, t::Int=1, σ::Float64=1.0)
    X = minmax_normalization(X)

    sumX = sum(X.^ 2, 1)
    K = exp(( sumX' .+ sumX .- 2*At_mul_B(X,X) ) ./ (2*sigma^2))

    p = sum(K, 1)'
    K ./= ((p * p') .^ t)
    p = sqrt(sum(K, 1))'
    K ./= (p * p')

    U, S, V = svd(K, thin=false)
    U ./= U[:,1]
    Y = U[:,2:(d+1)]

    return DiffMap(t, σ, K, Y')
end
