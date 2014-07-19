# Laplacian Eigenmaps
# -------------------
# Laplacian Eigenmaps for Dimensionality Reduction and Data Representation,
# M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396

immutable Eigenmap <: Results
    res::DistSpectralResults
    σ::Float64
    Eigenmap(proj::Matrix{Float64}, λ::Vector{Float64}, k::Int, lcc::Vector{Int},
        σ::Float64) = new(DistSpectralResults(proj, λ, k, lcc), σ)
end

## properties
indim(M::Eigenmap) = indim(M.res)
outdim(M::Eigenmap) = outdim(M.res)
projection(M::Eigenmap) = projection(M.res)
eigenvalues(M::Eigenmap) = eigenvalues(M.res)
nneighbors(M::Eigenmap) = nneighbors(M.res)
ccomponent(M::Eigenmap) = ccomponent(M.res)
variance(M::Eigenmap) = M.σ

function Base.show(io::IO, M::Eigenmap)
    println(io, "σ: $(M.σ)")
    show(io, M.res)
end

function lem{T}(X::AbstractMatrix{T}; d::Int=2, k::Int=12, σ::Float64=1.0)
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
    C = length(CC) == 1 ? CC[1] : CC[indmax(map(size, CC))]
    #W = W[C,C]

    # Compute weights
    W[W .> eps(T)] = exp(-W[W .> eps(T)] ./ (2*σ^2))
    D = diagm(sum(W,2)[:])
    L = D - W

    # Build eigenmaps
    λ, U = eig(L, D)
    λ = real(λ)[2:(d+1)]
    Y = real(U)[:,2:(d+1)]

    return Eigenmap(Y', λ, k, C, σ)
end
