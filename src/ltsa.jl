# Local Tangent Space Alignment (LTSA)
# ---------------------------
# Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment,
# Zhang, Zhenyue; Hongyuan Zha (2004),  SIAM Journal on Scientific Computing 26 (1): 313–338.
# doi:10.1137/s1064827502419154.

#### LTSA type
struct LTSA{T <: AbstractFloat} <: SpectralResult
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}

    LTSA{T}(k::Int, λ::AbstractVector{T}, proj::Projection{T}) where T = new(k, λ, proj)
end

## properties
outdim(M::LTSA) = size(M.proj, 1)
projection(M::LTSA) = M.proj

eigvals(M::LTSA) = M.λ
neighbors(M::LTSA) = M.k

## show & dump
function show(io::IO, M::LTSA)
    print(io, "LTSA(outdim = $(outdim(M)), neighbors = $(neighbors(M)))")
end

function dump(io::IO, M::LTSA)
    show(io, M)
    println(io, "eigenvalues: ")
    Base.showarray(io, transpose(M.λ), header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function transform(::Type{LTSA}, X::DenseMatrix{T}; d::Int=2, k::Int=12) where T<:AbstractFloat
    n = size(X, 2)

    # Construct NN graph
    D, I = find_nn(X, k)

    B = spzeros(n,n)
    for i=1:n
        # re-center points in neighborhood
        μ = mean(X[:,I[:,i]], dims=2)
        δ_x = X[:,I[:,i]] .- μ

        # Compute orthogonal basis H of θ'
        θ_t = svd(δ_x).V[:,1:d]

        # Construct alignment matrix
        G = hcat(ones(k)./sqrt(k), θ_t)
        B[I[:,i], I[:,i]] =  B[I[:,i], I[:,i]] + Matrix{Float64}(LinearAlgebra.I, k, k) - G*transpose(G)
    end

    # Align global coordinates
    λ, V = decompose(B, d)
    return LTSA{T}(k, λ, transpose(V))
end
