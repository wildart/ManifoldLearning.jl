# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

#### Isomap type
immutable Isomap <: SpectralResult
    k::Int
    λ::Vector{Float64}
    proj::Projection
    component::Vector{Int}

    Isomap(k::Int, λ::Vector{Float64}, proj::Projection) = new(k, λ, proj)
    Isomap(k::Int, λ::Vector{Float64}, proj::Projection, cc::Vector{Int}) = new(k, λ, proj, cc)
end

## properties
indim(M::Isomap) = size(M.proj, 1)
outdim(M::Isomap) = size(M.proj, 2)
projection(M::Isomap) = M.proj

eigvals(M::Isomap) = M.λ
nneighbors(M::Isomap) = M.k
ccomponent(M::Isomap) = M.component

## show & dump
function show(io::IO, M::Isomap)
    print(io, "Isomap(indim = $(indim(M)), outdim = $(outdim(M)), nneighbors = $(nneighbors(M)))")
end

function dump(io::IO, M::Isomap)
    show(io, M)
    # try # Print largest connected component
    #     lcc = ccomponent(M)
    #     println(io, "connected component: ")
    #     Base.showarray(io, lcc', header=false, repr=false)
    #     println(io)
    # end
    println(io, "eigenvalues: ")
    Base.showarray(io, M.λ', header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function fit(::Type{Isomap}, X::DenseMatrix{Float64}; d::Int=2, k::Int=12)
    # Construct NN graph
    D, E = find_nn(X, k)

    # Select largest connected component
    CC = components(E)
    if length(CC) == 1
        C = CC[1]
        Dc = D
        Ec = E
    else
        C = CC[indmax(map(size, CC))]
        Dc = D[:,C]

        # renumber edges
        R = Dict(C, 1:length(C))
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
    M = DD.^2
    B = (M .- sum(M, 1) ./ n .- sum(M, 2) ./ n .+ sum(M) ./ (n^2)) .* -0.5

    # Compute embedding
    λ, U = eig(B)
    indices = find(!(imag(λ) .< 0.0) .* !(imag(λ) .> 0.0) .* real(λ) .> 0)[1:d]
    λ = λ[indices]
    U = U[:, indices]
    Y = real(U .* sqrt(λ)')

    return Isomap(k, real(λ), Y', C)
end