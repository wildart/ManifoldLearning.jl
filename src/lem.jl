# Laplacian Eigenmaps
# -------------------
# Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
# M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396

#### LEM type
struct LEM{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    λ::AbstractVector{T}
    t::T
    proj::Projection{T}
    component::Vector{Int}

    LEM{T}(k::Int, λ::AbstractVector{T}, t::T, proj::Projection{T})  where T = new(k, λ, t, proj)
    LEM{T}(k::Int, λ::AbstractVector{T}, t::Float64, proj::Projection{T}, cc::Vector{Int})  where T = new(k, λ, t, proj, cc)
end

## properties
outdim(R::LEM) = size(R.proj, 1)
eigvals(R::LEM) = R.λ
neighbors(R::LEM) = R.k
vertices(R::LEM) = R.component

## show
summary(io::IO, R::LEM) = print(io, "Laplacian Eigenmaps(outdim = $(outdim(R)), neighbors = $(neighbors(R)), t = $(R.t))")

## interface functions
function fit(::Type{LEM}, X::AbstractMatrix{T}; maxoutdim::Int=2, k::Int=12, t::Real=1.0, knn=knn) where {T<:Real}
    # Construct NN graph
    D, E = knn(X, k)
    G, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))

    # Compute weights
    W = weights(G)
    W .^= 2
    W ./= maximum(W)

    W[W .> eps(T)] = exp.(-W[W .> eps(T)] ./ convert(T,t))
    D = diagm(0 => sum(W, dims=2)[:])
    L = D - W

    λ, V = decompose(L, D, maxoutdim)
    return LEM{T}(k, λ, t, transpose(V), C)
end

transform(R::LEM) = R.proj
