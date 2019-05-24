# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

#### Isomap type
struct Isomap{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    model::KernelPCA
    component::AbstractVector{Int}

    Isomap{T}(k::Int, model::KernelPCA) where T = new(k, model)
    Isomap{T}(k::Int, model::KernelPCA, cc::AbstractVector{Int}) where T = new(k, model, cc)
end

## properties
outdim(R::Isomap) = outdim(R.model)
eigvals(R::Isomap) = principalvars(R.model)
neighbors(R::Isomap) = R.k
vertices(R::Isomap) = R.component
transform(R::Isomap) = transform(R.model)

## show
summary(io::IO, R::Isomap) = print(io, "Isomap(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")
function show(io::IO, R::Isomap)
    summary(io, R)
    if !get(io, :short, true)
        io = IOContext(io, :limit=>true)
        println(io)
        println(io, "connected component: ")
        Base.show_vector(io, vertices(R))
        println(io)
        println(io, "eigenvalues: ")
        Base.show_vector(io, eigvals(R))
        println(io)
        println(io, "projection:")
        Base.print_matrix(io, transform(R), "[", ",","]")
    end
end

## interface functions
function fit(::Type{Isomap}, X::AbstractMatrix{T}; k::Int=12, maxoutdim::Int=2) where {T<:Real}
    # Construct NN graph
    D, E = find_nn(X, k)
    G, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))

    # Compute shortest path for every point
    n = length(C)
    DD = zeros(T, n, n)
    for i in 1:n
        dj = dijkstra_shortest_paths(G, i)
        DD[i,:] = dj.dists
    end

    broadcast!(x->-x*x/2, DD, DD)
    broadcast!((x,y)->(x+y)/2, DD, DD, DD') # remove roundoff error
    M = fit(KernelPCA, DD, kernel=nothing, maxoutdim=maxoutdim)

    return Isomap{T}(k, M, C)
end

@deprecate transform(Isomap, X; k=k, d=d) fit(Isomap, X; k=k, maxoutdim=d)
@deprecate projection(R::Isomap) transform(R)
