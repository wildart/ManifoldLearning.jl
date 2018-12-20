# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

#### Isomap type
struct Isomap{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    proj::Projection{T}
    component::AbstractVector{Int}

    Isomap{T}(k::Int, proj::Projection{T}) where T = new(k, proj)
    Isomap{T}(k::Int, proj::Projection{T}, cc::AbstractVector{Int}) where T = new(k, proj, cc)
end

## properties
outdim(M::Isomap) = size(M.proj, 1)
projection(M::Isomap) = M.proj

neighbors(M::Isomap) = M.k
vertices(M::Isomap) = M.component

## show & dump
function show(io::IO, M::Isomap)
    print(io, "Isomap(outdim = $(outdim(M)), neighbors = $(neighbors(M)))")
end

function dump(io::IO, M::Isomap)
    show(io, M)
    # try # Print largest connected component
    #     lcc = ccomponent(M)
    #     println(io, "connected component: ")
    #     Base.showarray(io, transpose(lcc), header=false, repr=false)
    #     println(io)
    # end
    println(io, "eigenvalues: ")
    Base.showarray(io, transpose(M.λ), header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function transform(::Type{Isomap}, X::AbstractMatrix{T}; k::Int=12, d::Int=2) where {T<:Real}
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

    # Perform MDS
    Y = classical_mds(DD, d)

    return Isomap{T}(k, Y, C)
end
