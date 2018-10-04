# Laplacian Eigenmaps
# -------------------
# Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
# M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396

#### LEM type
struct LEM{T <: AbstractFloat} <: SpectralResult
    k::Int
    λ::AbstractVector{T}
    t::T
    proj::Projection{T}
    component::Vector{Int}

    LEM{T}(k::Int, λ::AbstractVector{T}, t::T, proj::Projection{T})  where T = new(k, λ, t, proj)
    LEM{T}(k::Int, λ::AbstractVector{T}, t::Float64, proj::Projection{T}, cc::Vector{Int})  where T = new(k, λ, t, proj, cc)
end

## properties
outdim(M::LEM) = size(M.proj, 1)
projection(M::LEM) = M.proj

eigvals(M::LEM) = M.λ
neighbors(M::LEM) = M.k
ccomponent(M::LEM) = M.component

## show & dump
function show(io::IO, M::LEM)
    print(io, "Laplacian Eigenmaps(outdim = $(outdim(M)), neighbors = $(neighbors(M)), t = $(M.t))")
end

function dump(io::IO, M::LEM)
    show(io, M)
    println(io, "eigenvalues: ")
    Base.showarray(io, transpose(eigvals(M)), header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, projection(M), header=false, repr=false)
end

## interface functions
function transform(::Type{LEM}, X::DenseMatrix{T}; d::Int=2, k::Int=12, t::T=1.0) where T<:AbstractFloat
    n = size(X, 2)

    # Construct NN graph
    D, E = find_nn(X, k)

    W = zeros(T,n,n)
    for i = 1 : n
        jj = E[:, i]
        W[i,jj] = D[:, i]
    end
    W .^= 2
    W ./= maximum(W)

    # Select largest connected component
    CC = components(E)
    C = length(CC) == 1 ? CC[1] : CC[argmax(map(size, CC))]

    # Compute weights
    W[W .> eps(T)] = exp.(-W[W .> eps(T)] ./ t)
    D = diagm(0 => sum(W, dims=2)[:])
    L = D - W

    λ, V = decompose(L, D, d)
    return LEM{T}(k, λ, t, transpose(V), C)
end
