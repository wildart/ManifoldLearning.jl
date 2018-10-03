# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

#### Isomap type
struct Isomap{T <: AbstractFloat} <: SpectralResult
    k::Int
    proj::Projection{T}
    component::AbstractVector{Int}

    Isomap{T}(k::Int, proj::Projection{T})  where T = new(k, proj)
    Isomap{T}(k::Int, proj::Projection{T}, cc::AbstractVector{Int})  where T = new(k, proj, cc)
end

## properties
outdim(M::Isomap) = size(M.proj, 1)
projection(M::Isomap) = M.proj

neighbors(M::Isomap) = M.k
ccomponent(M::Isomap) = M.component

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
function transform(::Type{Isomap}, X::DenseMatrix{T};
                   k::Int=12, d::Int=2) where T <: AbstractFloat
    # Construct NN graph
    D, E = find_nn(X, k, excluding=true)

    # Select largest connected component
    CC = components(E)
    if length(CC) == 1
        C = CC[1]
        Dc = D
        Ec = E
    else
        C = CC[argmax(map(size, CC))]
        Dc = D[:,C]

        # renumber edges
        R = Dict(zip(C, 1:length(C)))
        Ec = zeros(Int,k,length(C))
        for i = 1 : length(C)
            Ec[:,i] = map(i->get(R,i,0), E[:,C[i]])
        end
    end

    # Compute shortest path for every point
    n = size(Dc,2)
    DD = zeros(n, n)
    for i=1:n
        P, PD = dijkstra(Dc, Ec, i)
        DD[i,:] = PD
    end

    # Perform MDS
    Y = classical_mds(DD, d)

    return Isomap{T}(k, Y, C)
end
