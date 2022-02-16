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
    nearestneighbors::NN
    component::AbstractVector{Int}
    model::KernelPCA
end
Isomap(d::Int, nn::NN, model::KernelPCA) where {NN <: AbstractNearestNeighbors} = Isomap(nn, model, Int[])

## properties
size(R::Isomap) = (R.d, size(R.model)[2])
eigvals(R::Isomap) = eigvals(R.model)
neighbors(R::Isomap) = R.nearestneighbors.k
vertices(R::Isomap) = R.component

## show
function summary(io::IO, R::Isomap{T}) where {T}
    id, od = size(R)
    print(io, "Isomap{$T}(indim = $id, outdim = $od, neighbors = $(neighbors(R)))")
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
    NN = fit(nntype, X, k)
    D, E = knn(NN, X)
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

    return Isomap{nntype}(d, NN, C, M)
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
    D, E = knn(R.nearestneighbors, X, self = true)
    DD = gram2dmat(R.model.X)

    G = zeros(size(R.model.X,2), n)
    for i in 1:n
        G[:,i] = minimum(DD[:,E[:,i]] .+ D[:,i]', dims=2)
    end

    broadcast!(x->-x*x/2, G, G)
    transform!(R.model.center, G)
    return projection(R.model)'*G'
end
