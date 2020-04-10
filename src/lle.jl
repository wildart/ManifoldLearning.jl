# Locally Linear Embedding (LLE)
# ------------------------
# Nonlinear dimensionality reduction by locally linear embedding,
# Roweis, S. & Saul, L., Science 290:2323 (2000)

"""

    LLE{NN <: AbstractNearestNeighbors, T <: Real} <: AbstractDimensionalityReduction

The `LLE` type represents locally linear embedding model with `T` type data constructed with a help of the `NN` nearest neighbor algorithm.
"""
struct LLE{NN <: AbstractNearestNeighbors, T <: Real} <: AbstractDimensionalityReduction
    nearestneighbors::NN
    component::AbstractVector{Int}
    λ::AbstractVector{T}
    proj::Projection{T}
end

## properties
outdim(R::LLE) = size(R.proj, 1)
eigvals(R::LLE) = R.λ
neighbors(R::LLE) = R.nearestneighbors.k
vertices(R::LLE) = R.component

## show
summary(io::IO, R::LLE) = print(io, "LLE(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
"""
    fit(LLE, data; k=12, maxoutdim=2, nntype=BruteForce, tol=1e-5)

Fit a locally linear embedding model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: an type of the nearest neighbor construction (derived from `AbstractNearestNeighbors`)
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
    NN = fit(nntype, X, k)
    D, E = knn(NN, X)
    _, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))

    X = @view X[:, C]
    n = length(C)

    # Correct indexes of neighbors if more then one connected component
    Ec = E
    if size(E,2) != n
        R = Dict(zip(C, collect(1:n)))
        Ec = zeros(Int,k,n)
        for i in 1 : n
            Ec[:,i] = map(j->get(R,j,C[i]), E[:,C[i]])
        end
    end

    if k > maxoutdim
        @warn("k > maxoutdim: regularization will be used")
    else
        tol = 0
    end

    # Reconstruct weights and compute embedding:
    # M = (I - w)'(I - w) = I - w'I - Iw + w'w
    M = spdiagm(0 => fill(one(T), n))
    Ones = fill(one(T), k, 1)
    for i in 1 : n
        J = Ec[:,i]
        Z = view(X, :, J) .- view(X, :, i)
        G = transpose(Z)*Z
        G += I * tol # regularize
        w = vec(G \ Ones)
        w ./= sum(w)
        ww = w*transpose(w)
        for (l, j) in enumerate(J)
            M[i,j] -= w[l]
            M[j,i] -= w[l]
            for (m, jj) in enumerate(J)
                M[j,jj] = ww[l,m]
            end
        end
    end

    λ, V = decompose(M, maxoutdim)
    return LLE{nntype, T}(NN, C, λ, rmul!(transpose(V), sqrt(n)))
end

"""
    transform(R::LLE)

Transforms the data fitted to the LLE model `R` into a reduced space representation.
"""
transform(R::LLE) = R.proj
