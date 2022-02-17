# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

"""

    Isomap{NN <: AbstractNearestNeighbors} <: AbstractDimensionalityReduction

The `Isomap` type represents an isometric mapping model constructed with a help of the `NN` nearest neighbor algorithm.
"""
struct Isomap{NN <: AbstractNearestNeighbors} <: NonlinearDimensionalityReduction
    d::Int
    k::Int
    model::KernelPCA
    nearestneighbors::NN
    component::AbstractVector{Int}
end

## properties
size(R::Isomap) = (R.d, size(R.model)[2])
eigvals(R::Isomap) = eigvals(R.model)
neighbors(R::Isomap) = R.k
vertices(R::Isomap) = R.component

## show
function summary(io::IO, R::Isomap)
    id, od = size(R)
    print(io, "Isomap{$(R.nearestneighbors)}(indim = $id, outdim = $od, neighbors = $(neighbors(R)))")
end

## interface functions
"""
    fit(Isomap, data; k=12, maxoutdim=2, nntype=BruteForce)

Fit an isometric mapping model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `k`: a number of nearest neighbors for construction of local subspace representation
* `maxoutdim`: a dimension of the reduced space.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)

# Examples
```julia
M = fit(Isomap, rand(3,100)) # construct Isomap model
R = predict(M)               # perform dimensionality reduction
```
"""
function fit(::Type{Isomap}, X::AbstractMatrix{T};
             k::Int=12, maxoutdim::Int=2, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    d, n = size(X)
    NN = fit(nntype, X)
    D, E = knn(NN, X, k)
    A = adjmat(D,E)
    G, C = largest_component(SimpleGraph(A))

    # Compute shortest path for every point
    n = length(C)
    DD = zeros(T, n, n)
    for i in 1:n
        dj = dijkstra_shortest_paths(G, i, A)
        DD[i,:] .= dj.dists
    end

    M = fit(KernelPCA, dmat2gram(DD), kernel=nothing, maxoutdim=maxoutdim)

    return Isomap{nntype}(d, k, M, NN, C)
end

"""
    predict(R::Isomap)

Transforms the data fitted to the Isomap model `R` into a reduced space representation.
"""
predict(R::Isomap) = predict(R.model)

"""
    predict(R::Isomap, X::AbstractVecOrMat)

Returns a transformed out-of-sample data `X` given the Isomap model `R` into a reduced space representation.
"""
function predict(R::Isomap, X::AbstractVecOrMat{T}) where {T<:Real}
    n = size(X,2)
    D, E = knn(R.nearestneighbors, X, R.k, self = true)
    DD = gram2dmat(R.model.X)

    G = zeros(size(R.model.X,2), n)
    for i in 1:n
        G[:,i] = minimum(DD[:,E[:,i]] .+ D[:,i]', dims=2)
    end

    broadcast!(x->-x*x/2, G, G)
    transform!(R.model.center, G)
    return projection(R.model)'*G'
end
