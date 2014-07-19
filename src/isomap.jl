# Isomap
# ------
# A Global Geometric Framework for Nonlinear Dimensionality Reduction,
# J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000

typealias Isomap DistSpectralResults

function isomap(X::Matrix; d::Int=2, k::Int=12)
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

    return Isomap(Y', real(λ), k, C)
end