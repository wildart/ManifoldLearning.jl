# Local Tangent Space Alignment (LTSA)
# ---------------------------
# Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment,
# Zhang, Zhenyue; Hongyuan Zha (2004),  SIAM Journal on Scientific Computing 26 (1): 313–338.
# doi:10.1137/s1064827502419154.

#### LTSA type
immutable LTSA <: SpectralResult
    k::Int
    λ::Vector{Float64}
    proj::Projection

    LTSA(k::Int, λ::Vector{Float64}, proj::Projection) = new(k, λ, proj)
end

## properties
indim(M::LTSA) = size(M.proj, 1)
outdim(M::LTSA) = size(M.proj, 2)
projection(M::LTSA) = M.proj

eigvals(M::LTSA) = M.λ
nneighbors(M::LTSA) = M.k

## show & dump
function show(io::IO, M::LTSA)
    print(io, "LTSA(indim = $(indim(M)), outdim = $(outdim(M)), nneighbors = $(nneighbors(M)))")
end

function dump(io::IO, M::LTSA)
    show(io, M)
    println(io, "eigenvalues: ")
    Base.showarray(io, M.λ', header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function fit(::Type{LTSA}, X::DenseMatrix{Float64}; d::Int=2, k::Int=12)
    n = size(X, 2)

    # Construct NN graph
    D, I = find_nn(X, k)

    B = spzeros(n,n)
    for i=1:n
        # re-center points in neighborhood
        μ = mean(X[:,I[:,i]],2)
        δ_x = X[:,I[:,i]] .- μ

        # Compute orthogonal basis H of θ'
        θ_t = svdfact(δ_x)[:V][:,1:d]
        #θ_t = (svdfact(δ_x)[:U][:,1:d]'*δ_x)'
        H = full(qrfact(θ_t)[:Q])

        # Construct alignment matrix
        G = hcat(ones(k)./sqrt(k), H)
        B[I[:,i], I[:,i]] =  eye(k) - G*G'
    end

    # Align global coordinates
    λ, V = decompose(B, d)
    return LLE(k, λ, V')
end
