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
    k::Real
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
    msg = isinteger(R.k) ? "neighbors" : "epsilon"
    print(io, "LTSA{$(R.nearestneighbors)}(indim = $id, outdim = $od, neighbors = $(R.k))")
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
             k::Real=12, maxoutdim::Int=2, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    d, n = size(X)
    NN = fit(nntype, X)
    E, _ = adjacency_list(NN, X, k)
    _, C = largest_component(SimpleGraph(n, E))

    # Correct indexes of neighbors if more then one connected component
    fixindex = length(C) < n
    if fixindex
        n = length(C)
        R = Dict(zip(C, collect(1:n)))
    end

    B = spzeros(T,n,n)
    for i in C
        NI = E[i] # neighbor's indexes

        # fix indexes for connected components
        NIfix, NIcc = if fixindex # fix index
            JJ = [i for i in NI if i ∈ C] # select points that are in CC
            KK = [R[i] for i in JJ if haskey(R, i)] # convert NI to CC index
            JJ, KK
        else
            NI, NI
        end
        l = length(NIfix)
        l == 0 && continue # skip

        # re-center points in neighborhood
        VX = view(X, :, NIfix)
        μ = mean(VX, dims=2)
        δ_x = VX .- μ

        # Compute orthogonal basis H of θ'
        θ_t = view(svd(δ_x).V, :, 1:maxoutdim)

        # Construct alignment matrix
        S = ones(l)./sqrt(l)
        G = hcat(S, θ_t)
        B[NIcc, NIcc] .+= Diagonal(fill(one(T), l)) .- G*transpose(G)
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

