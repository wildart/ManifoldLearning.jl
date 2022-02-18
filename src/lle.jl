# Locally Linear Embedding (LLE)
# ------------------------
# Nonlinear dimensionality reduction by locally linear embedding,
# Roweis, S. & Saul, L., Science 290:2323 (2000)

"""

    LLE{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction

The `LLE` type represents a locally linear embedding model constructed for `T` type data constructed with a help of the `NN` nearest neighbor algorithm.
"""
struct LLE{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction
    d::Int
    k::Real
    λ::AbstractVector{T}
    proj::Projection{T}
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
size(R::LLE) = (R.d, size(R.proj, 1))
eigvals(R::LLE) = R.λ
neighbors(R::LLE) = R.k
vertices(R::LLE) = R.component

## show
function summary(io::IO, R::LLE)
    id, od = size(R)
    msg = isinteger(R.k) ? "neighbors" : "epsilon"
    print(io, "LLE{$(R.nearestneighbors)}(indim = $id, outdim = $od, $msg = $(R.k))")
end

## interface functions
"""
    fit(LLE, data; k=12, maxoutdim=2, nntype=BruteForce, tol=1e-5)

Fit a locally linear embedding model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)
* `tol`: an algorithm regularization tolerance

# Examples
```julia
M = fit(LLE, rand(3,100)) # construct LLE model
R = transform(M)          # perform dimensionality reduction
```
"""
function fit(::Type{LLE}, X::AbstractMatrix{T};
             k::Int=12, maxoutdim::Int=2, nntype=BruteForce, tol::Real=1e-5) where {T<:Real}
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

    if k > d
        @warn("k > $d: regularization will be used")
    else
        tol = 0
    end

    # Reconstruct weights and compute embedding:
    M = spdiagm(0 => fill(one(T), n))
    #W = spzeros(T, n, n)
    O = fill(one(T), k, 1)
    for i in C
        NI = E[i] # neighbor's indexes

        # fix indexes for connected components
        NIfix, NIcc, j = if fixindex # fix index
            JJ = [i for i in NI if i ∈ C] # select points that are in CC
            KK = [R[i] for i in JJ if haskey(R, i)] # convert NI to CC index
            JJ, KK, R[i]
        else
            NI, NI, i
        end
        l = length(NIfix)
        l == 0 && continue # skip

        # centering neighborhood of point xᵢ
        zᵢ = view(X, :, NIfix) .- view(X, :, i)

        # calculate weights: wᵢ = (Gᵢ + αI)⁻¹1
        G = zᵢ'zᵢ
        w = (G  + tol * I) \ fill(one(T), l, 1)
        w ./= sum(w)

        # M = (I - w)'(I - w) = I - w'I - Iw + w'w
        M[NIcc,j] .-= w
        M[j,NIcc] .-= w
        M[NIcc,NIcc] .+= w*w'

        #W[NI, i] .= w
    end
    #@assert all(sum(W, dims=1).-1 .< tol) "Weights are not normalized"
    #M = (I-W)*(I-W)'

    λ, V = decompose(M, maxoutdim)
    return LLE{nntype, T}(d, k, λ, transpose(V) .* convert(T, sqrt(n)), NN, C)
end

"""
    predict(R::LLE)

Transforms the data fitted to the LLE model `R` into a reduced space representation.
"""
predict(R::LLE) = R.proj

