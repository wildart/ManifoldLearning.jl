# Local Tangent Space Alignment (LTSA)
# ---------------------------
# Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment,
# Zhang, Zhenyue; Hongyuan Zha (2004),  SIAM Journal on Scientific Computing 26 (1): 313–338.
# doi:10.1137/s1064827502419154.

"""
    LTSA{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction

The `LTSA` type represents a local tangent space alignment model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct LTSA{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction
    d::Int
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
size(R::LTSA) = (R.d, size(R.proj, 1))
eigvals(R::LTSA) = R.λ
neighbors(R::LTSA) = R.k
vertices(R::LTSA) = R.component

## show
function summary(io::IO, R::LTSA)
    id, od = size(R)
    print(io, "LTSA{$(R.nearestneighbors)}(indim = $id, outdim = $od, neighbors = $(neighbors(R)))")
end

## interface functions
"""
    fit(LTSA, data; k=12, maxoutdim=2, nntype=BruteForce)

Fit a local tangent space alignment model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)

# Examples
```julia
M = fit(LTSA, rand(3,100)) # construct LTSA model
R = transform(M)           # perform dimensionality reduction
```
"""
function fit(::Type{LTSA}, X::AbstractMatrix{T};
        k::Int=12, maxoutdim::Int=2, ɛ::Real=1.0, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    NN = fit(nntype, X)
    D, E = knn(NN, X, k)
    A = adjmat(D,E)
    G, C = largest_component(SimpleGraph(A))
    XX = @view X[:, C]
    d, n = size(X)

    S = ones(k)./sqrt(k)
    B = spzeros(T,n,n)
    for i=1:n
        II = @view E[:,i]
        VX = view(XX, :, II)

        # re-center points in neighborhood
        μ = mean(VX, dims=2)
        δ_x = VX .- μ

        # Compute orthogonal basis H of θ'
        θ_t = svd(δ_x).V[:,1:maxoutdim]

        # Construct alignment matrix
        G = hcat(S, θ_t)
        B[II, II] .+= Diagonal(fill(one(T), k)) .- G*transpose(G)
    end

    # Align global coordinates
    λ, V = decompose(B, maxoutdim)
    return LTSA{nntype, T}(d, k, λ, transpose(V), NN, C)
end

"""
    predict(R::LTSA)

Transforms the data fitted to the local tangent space alignment model `R` into a reduced space representation.
"""
predict(R::LTSA) = R.proj

