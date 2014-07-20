using Base.Collections

# Generate all combinations of "n" elements from a given iterable object
function combn(v::Vector, n::Int)
    nv = length(v)
    A = [0:nv^n-1]+(1/2)
    B = [nv.^[1-n:0.]]
    IND = int(rem(floor(A * B'),nv) + 1)
    v[IND]
end

# Generate k-nearest neighborhood graph with distances
function find_nn{T}(X::AbstractMatrix{T}, k::Int=12)
    m, n = size(X)
    r = Array(T, (n, n))
    d = Array(T, k, n)
    e = Array(Int, k, n)

    At_mul_B!(r, X, X)
    sa2 = sum(X.^2, 1)

    for j = 1 : n
        for i = 1 : j-1
            @inbounds r[i,j] = r[j,i]
        end
        @inbounds r[j,j] = 0
        for i = j+1 : n
            @inbounds v = sa2[i] + sa2[j] - 2 * r[i,j]
            @inbounds r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
        e[:, j] = sortperm(r[:,j])[2:k+1]
        d[:, j] = r[e[:, j],j]
    end

    return (d, e)
end

# find connected components for undirected graph
function components(E::AbstractMatrix{Int})
    m, n = size(E)
    cmap = zeros(Int, n)
    cc = Array(Vector{Int}, 0)
    queue = Int[]

    for v in 1 : n
        if cmap[v] == 0
            # Start BFS with coloring
            c = Int[]
            push!(queue, v)
            while !isempty(queue)
                w = shift!(queue)
                for k in E[:, w]
                    if cmap[k] == 0
                        cmap[k] = 1
                        push!(queue, k)
                    end
                end
                cmap[w] = 2
                push!(c, w)
            end
            # Save component
            push!(cc, c)
        end
    end

    return cc
end

# Dijkstra's algorithm for single-source shortest path
function dijkstra{T}(D::AbstractMatrix{T}, E::AbstractMatrix{Int}, src)
    m, n = size(D)
    path = zeros(Int, n)
    dist = fill(Inf, n)
    dist[src] = 0

    q = PriorityQueue(1:n, dist)
    while !isempty(q)
        u = dequeue!(q)
        for i in 1:m
            v = E[i,u]
            alt = dist[u] + D[i,u]
            if haskey(q, v) && alt < q[v]
                dist[v] = q[v] = alt
                path[v] = u
            end
        end
    end

    return (path, dist)
end

# find strongly connected components for directed graph
function scomponents(E::AbstractMatrix{Int})

    function tarjan(v::Int)
        global E, I, index, lowlink, stack, components
        index[v] = I
        lowlink[v] = I
        I += 1
        push!(stack, v)

        for w in E[:,v]
            if index[w] == -1
                tarjan(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elseif w in stack
                lowlink[v] = min(lowlink[v], lowlink[w])
            end
        end

        if lowlink[v] == index[v]
            component = Int[]
            while true
                w = pop!(stack)
                push!(component, w)
                if w == v
                    break
                end
            end
            push!(components, component)
        end
    end

    m, n = size(E)
    I = 0
    index = fill(-1, n)
    lowlink = fill(-1, n)
    stack = Int[]
    components = Array{Int,1}[]

    for v = 1 : n
        if index[v] == -1
            tarjan(v)
        end
    end

    return components
end

function minmax_normalization(X::Matrix)
    X .-= minimum(X)
    X ./ maximum(X)
end

# Swiss roll dataset
function swiss_roll(n::Int = 1000, noise::Float64=0.05)
    t = (3 * pi / 2) * (1 .+ 2 * rand(n, 1))
    height = 30 * rand(n, 1)
    X = [t .* cos(t) height t .* sin(t)] + noise * randn(n, 3)
    labels = vec(int(rem(sum([round(t / 2) round(height / 12)], 2), 2)))
    return X', labels
end

function decompose{T}(M::AbstractMatrix{T}, d::Int)
    W = isa(M, AbstractSparseArray) ? Symmetric(full(M)) : Symmetric(M)
    F = eigfact!(W)
    idx = sortperm(F[:values])[2:d+1]
    return F[:values][idx], F[:vectors][:,idx]
end

function decompose{T}(A::AbstractMatrix{T}, B::AbstractMatrix{T}, d::Int)
    AA = isa(A, AbstractSparseArray) ? Symmetric(full(A)) : Symmetric(A)
    BB = isa(B, AbstractSparseArray) ? Symmetric(full(B)) : Symmetric(B)
    F = eigfact(AA, BB)
    idx = sortperm(F[:values])[2:d+1]
    return F[:values][idx], F[:vectors][:,idx]
end