# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

#### Isomap type
struct Isomap{NN <: AbstractNearestNeighbors} <: AbstractDimensionalityReduction
    nearestneighbors::NN
    model::KernelPCA
    component::AbstractVector{Int}

    Isomap{NN}(nn::NN, model::KernelPCA) where NN = new(nn, model)
    Isomap{NN}(nn::NN, model::KernelPCA, cc::AbstractVector{Int}) where NN = new(nn, model, cc)
end

## properties
outdim(R::Isomap) = outdim(R.model)
eigvals(R::Isomap) = principalvars(R.model)
neighbors(R::Isomap) = R.nearestneighbors.k
vertices(R::Isomap) = R.component

## show
summary(io::IO, R::Isomap{T}) where T =
    print(io, "Isomap{$T}(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
function fit(::Type{Isomap}, X::AbstractMatrix{T};
             k::Int=12, maxoutdim::Int=2, nntype=BruteForce) where {T<:Real}
    # Construct NN graph
    NN = fit(nntype, X, k)
    D, E = knn(NN, X)
    G, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))

    # Compute shortest path for every point
    n = length(C)
    DD = zeros(T, n, n)
    for i in 1:n
        dj = dijkstra_shortest_paths(G, i)
        DD[i,:] = dj.dists
    end

    M = fit(KernelPCA, dmat2gram(DD), kernel=nothing, maxoutdim=maxoutdim)

    return Isomap{nntype}(NN, M, C)
end

transform(R::Isomap) = transform(R.model)

function transform(R::Isomap, X::AbstractVecOrMat{T}) where {T<:Real}
    n = size(X,2)
    D, E = knn(R.nearestneighbors, X, self = true)
    DD = gram2dmat(R.model.X)
    G = zeros(size(R.model.X,2), n)
    for i in 1:n
        G[:,i] = minimum(DD[:,E[:,i]] .+ D[:,i]', dims=2)
    end
    broadcast!(x->-x*x/2, G, G)
    transform!(R.model.center, G)
    return projection(R.model)'*G
end
