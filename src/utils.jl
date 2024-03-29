"""
    knn(points::AbstractMatrix, k) -> distances, indices

Performs a lookup of the `k` nearest neigbours for each point in the `points`
dataset to the dataset itself, and returns distances to and indices of the neigbours.

*Note: Inefficient implementation that uses distance matrix. Not recomended for large datasets.*
"""
function knn(X::AbstractMatrix{T}, k::Int=12) where T<:Real
    m, n = size(X)
    @assert n > k "Number of observations must be more then $(k)"

    r = Array{T}(undef, (n, n))
    d = Array{T}(undef, k, n)
    e = Array{Int}(undef, k, n)

    mul!(r, transpose(X), X)
    sa2 = sum(X.^2, dims=1)

    @inbounds for j = 1 : n
        @inbounds for i = 1 : j-1
             r[i,j] = r[j,i]
        end
        r[j,j] = 0
        @inbounds for i = j+1 : n
            v = sa2[i] + sa2[j] - 2 * r[i,j]
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
        e[:, j] = sortperm(r[:,j])[2:k+1]
        d[:, j] = r[e[:, j],j]
    end

    return (d, e)
end

"""
    adjacency_matrix(A, W)

Returns a weighted adjacency matrix constructed from an adjacency list `A` and
weights `W`.
"""
function adjacency_matrix(A::AbstractArray{S},
                          W::AbstractArray{Q}) where {T<:Real,
                                                       S<:AbstractArray{<:Integer},
                                                       Q<:AbstractArray{T}}
    @assert length(A) == length(W) "Weights and edge matrix must be of the same size"
    n = length(A)
    M = spzeros(T, n, n)
    @inbounds for (j, (ii, ws)) in enumerate(zip(A, W))
        M[ii, j] = ws
        M[j, ii] = ws
    end
    return M
end

"""
    Laplacian(A)

Construct Laplacian matrix `L` from the adjacency matrix `A`, s.t. ``L = D - A``
where ``D_{i,i} = \\sum_j A_{ji}``.
"""
function Laplacian(A::AbstractMatrix)
    D = Diagonal(vec(sum(A, dims=1)))
    return D - A, D
end

"""
    normalize!(L, [D; α=1, norm=:sym])

Performs in-place normalization of the Laplacian `L` using the degree matrix `D`,
if provided, raised in a power `α`.

The `norm` parameter specifies normalization type:
- `:sym`: Laplacian `L` is symmetrically normalized, s.t. ``L_{sym} = D^{-\\alpha} L D^{-\\alpha}``.
- `:rw`: Laplacian `L` is random walk normalized, s.t. ``L_{rw} = D^{-\\alpha} L``.

where ``D`` is a diagonal matrix, s.t. ``D_{i,i} = \\sum_j L_{ji}``.
"""
function normalize!(L::AbstractMatrix, D=Diagonal(vec(sum(L, dims=1)));
                    α::Real=1.0, norm=:rw)
    D⁻¹ =  Diagonal(1 ./ diag(D).^α)
    if norm == :sym
        rmul!(lmul!(D⁻¹, L),D⁻¹)
    elseif norm == :rw
        lmul!(D⁻¹, L)
    else
        error("Uknown normalization: $norm")
    end
end

"Crate a graph with largest connected component of adjacency matrix `W`"
function largest_component(G)
    CC = connected_components(G)
    if length(CC) > 1
        @warn "Found $(length(CC)) connected components. Largest component is selected."
        C = CC[argmax(map(length, CC))]
        G = first(induced_subgraph(G, C))
    else
        C = first(CC)
    end
    return G, C
end

"""
    swiss_roll(n::Int, noise::Real=0.03, segments=1)

Generate a swiss roll dataset of `n` points with point coordinate `noise` variance, and partitioned on number of `segments`.
"""
function swiss_roll(n::Int = 1000, noise::Real=0.03; segments=1, hlims=(-10.0,10.0),
                    rng::AbstractRNG=default_rng())
    t = (3 * pi / 2) * (1 .+ 2 * rand(rng, n, 1))
    height = (hlims[2]-hlims[1]) * rand(rng, n, 1) .+ hlims[1]
    X = [t .* cos.(t) height t .* sin.(t)]
    X .+= noise * randn(rng, n, 3)
    mn,mx = extrema(t)
    labels = segments == 0 ? t : round.(Int, (t.-mn)./(mx-mn).*(segments-1))
    return collect(transpose(X)), labels
end

"""
    spirals(n::Int, noise::Real=0.03, segments=1)

Generate a spirals dataset of `n` points with point coordinate `noise` variance.
"""
function spirals(n::Int = 1000, noise::Real=0.03; segments=1,
                 rng::AbstractRNG=default_rng())
    t = collect(1:n) / n * 2π
    height = 30 * rand(rng, n, 1)
    X = [cos.(t).*(.5cos.(6t).+1) sin.(t).*(.4cos.(6t).+1) 0.4sin.(6t)] + noise * randn(n, 3)
    labels = segments == 0 ? t : vec(rem.(sum([round.(Int, t / 2) round.(Int, height / 12)], dims=2), 2))
    return collect(transpose(X)), labels
end

"""
    scurve(n::Int, noise::Real=0.03, segments=1)

Generate an S curve dataset of `n` points with point coordinate `noise` variance.
"""
function scurve(n::Int = 1000, noise::Real=0.03; segments=1,
                 rng::AbstractRNG=default_rng())
    t = 3π*(rand(rng, n) .- 0.5)
    x = sin.(t)
    y = 2rand(rng, n)
    z = sign.(t) .* (cos.(t) .- 1)
    height = 30 * rand(rng, n, 1)
    X = [x y z] + noise * randn(n, 3)
    mn,mx = extrema(t)
    labels = segments == 0 ? t : round.(Int, (t.-mn)./(mx-mn).*(segments-1))
    return collect(transpose(X)), labels
end

"""
    pairwise!(f, dest, x; skipdiagonal=false)

Stores in a vector `dest`, that is a packed upper triangular matrix representation,
holding the result of applying `f` to all possible pairs of entries in
iterators `x`.
"""
function pairwise!(f, dest::AbstractVector{T}, x;
                   skipdiagonal=false) where {T<:Real}
    # check sizes
    n = length(x)
    l = length(dest)
    nelem = (n*(n+(skipdiagonal ? -1 : 1)))>>1
    l < nelem && throw(ArgumentError("Not enough elements in `dest`. Must be at least $nelem"))

    k = 1
    @inbounds for (i, xi) in enumerate(x), (j, yj) in enumerate(x)
        (i > j || (skipdiagonal && i == j )) && continue
        dest[k] = f(xi, yj)
        k+=1
    end
    return dest
end

"""
    unpack(v [; skipdiagonal=false])

Return a symmetric matrix from from packed upper triangular matrix, stored as
vector `v`,
"""
function unpack(v::AbstractVector{T}; skipdiagonal=false) where {T}
    l = length(v)
    nelem = (sqrt(8l+1)+1)/2
    !isinteger(nelem) && throw(ArgumentError("Incorrect input size"))

    n = Int(nelem)-!skipdiagonal
    A = zeros(T, n, n)
    k=1
    for i in 1:(n-skipdiagonal), j in (i+skipdiagonal):n
        A[i,j] = v[k]
        k+=1
    end
    return Symmetric(A)
end

"""
    compact(A)

Return a packed upper triangular part of the symmetric matrix `A` as a vector.
"""
function compact(S::AbstractMatrix{T}) where {T}
    n = size(S,1)
    C = zeros(T, (n*(n-1))>>1)
    compact!(C, S)
end
function compact!(C::AbstractVector{T}, S::AbstractMatrix{T}) where {T}
    n = size(S,1)
    k = 1
    for i in 1:n-1
        for j in i+1:n
            C[k] = S[i,j]
            k+=1
        end
    end
    return C
end

"""
    sparse(E, W, n)

Construct a sparse weighted adjacency `(n,n)`-matrix from the adjacency list `E`
and weights `W`.
"""
function sparse(E::AbstractVector{TI}, W::AbstractVector{TV}, n::Integer;
                symmetric::Bool=true) where {TI<:AbstractVector{<:Integer},
                                             TV<:AbstractVector{<:Real}}
    # check sizes
    k, l = length(E), length(W)
    k != l && throw(ArgumentError("Incorrect input size"))
    # construct sp matrix
    A = spzeros(eltype(eltype(W)), n, n)
    @inbounds for (j,(es, ds)) in enumerate(zip(E, W))
        A[es, j] .= ds
        if symmetric
            A[j, es] .= ds
        end
    end
    return A
end

"Perform spectral decomposition for Ax=λI"
function decompose(M::AbstractMatrix{<:Real}, d::Int; rev=false, skipfirst=true)
    W = isa(M, AbstractSparseMatrix) ? Symmetric(Matrix(M)) : Symmetric(M)
    F = eigen!(W)
    rng = 1:d
    idx = sortperm(F.values, rev=rev)[skipfirst ? rng.+1 : rng]
    return F.values[idx], F.vectors[:,idx]
end

"Perform spectral decomposition for Ax=λB"
function decompose(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}, d::Int; rev=false)
    AA = isa(A, AbstractSparseMatrix) ? Symmetric(Matrix(A)) : Symmetric(A)
    BB = isa(B, AbstractSparseMatrix) ? Symmetric(Matrix(B)) : Symmetric(B)
    F = eigen(AA, BB)
    idx = sortperm(F.values, rev=rev)[2:d+1]
    return F.values[idx], F.vectors[:,idx]
end

