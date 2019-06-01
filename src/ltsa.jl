# Local Tangent Space Alignment (LTSA)
# ---------------------------
# Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment,
# Zhang, Zhenyue; Hongyuan Zha (2004),  SIAM Journal on Scientific Computing 26 (1): 313–338.
# doi:10.1137/s1064827502419154.

#### LTSA type
struct LTSA{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}

    LTSA{T}(k::Int, λ::AbstractVector{T}, proj::Projection{T}) where T = new(k, λ, proj)
end

## properties
outdim(R::LTSA) = size(R.proj, 1)
eigvals(R::LTSA) = R.λ
neighbors(R::LTSA) = R.k

## show
summary(io::IO, R::LTSA) = print(io, "LTSA(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
function fit(::Type{LTSA}, X::AbstractMatrix{T}; maxoutdim::Int=2, k::Int=12, knn=knn) where {T<:Real}
    n = size(X, 2)

    # Construct NN graph
    D, E = knn(X, k)
    S = ones(k)./sqrt(k)
    B = spzeros(T, n,n)
    for i=1:n
        II = @view E[:,i]

        # re-center points in neighborhood
        μ = mean(X[:, II], dims=2)
        δ_x = X[:, II] .- μ

        # Compute orthogonal basis H of θ'
        θ_t = svd(δ_x).V[:,1:maxoutdim]

        # Construct alignment matrix
        G = hcat(S, θ_t)
        B[II, II] .+= diagm(0 => fill(one(T), k)) .- G*transpose(G)
    end

    # Align global coordinates
    λ, V = decompose(B, maxoutdim)
    return LTSA{T}(k, λ, transpose(V))
end

transform(R::LTSA) = R.proj
