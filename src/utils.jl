import DataStructures: PriorityQueue, dequeue!

"Generate k-nearest neighborhood graph with distances"
function find_nn(X::AbstractMatrix{T}, k::Int=12; excluding=true) where T<:Real
    m, n = size(X)
    r = Array{T}(undef, (n, n))
    d = Array{T}(undef, excluding ? k : k+1, n)
    e = Array{Int}(undef, excluding ? k : k+1, n)

    mul!(r, transpose(X), X)
    sa2 = sum(X.^2, dims=1)
    idx_range = excluding ? (2:k+1) : (1:k+1)

    @inbounds for j = 1 : n
        @inbounds for i = 1 : j-1
             r[i,j] = r[j,i]
        end
        r[j,j] = 0
        @inbounds for i = j+1 : n
            v = sa2[i] + sa2[j] - 2 * r[i,j]
            r[i,j] = isnan(v) ? NaN : sqrt(max(v, 0.))
        end
        e[:, j] = sortperm(r[:,j])[idx_range]
        d[:, j] = r[e[:, j],j]
    end

    return (d, e)
end

"Find connected components for a undirected graph given its adjacency matrix"
function components(E::AbstractMatrix{Int})
    m, n = size(E)
    cmap = zeros(Int, n)
    cc = Vector{Vector{Int}}(undef, 0)
    queue = Int[]

    for v in 1 : n
        if cmap[v] == 0
            # Start BFS with coloring
            c = Int[]
            push!(queue, v)
            while !isempty(queue)
                w = popfirst!(queue)
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

"Dijkstra's algorithm for single-source shortest path"
function dijkstra(D::AbstractMatrix{T}, E::AbstractMatrix{Int}, src::Int, dst::Int =-1) where T<:Real
    m, n = size(D)
    path = zeros(Int, n)
    dist = fill(Inf, n)
    dist[src] = 0

    q = PriorityQueue(zip(1:n, dist))
    while !isempty(q)
        u = dequeue!(q)
        if u == dst
            break
        end
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

"Find strongly connected components for undirected graph given its adjacency matrix"
function scomponents(E::AbstractMatrix{Int})

    function tarjan(v::Int)
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

"Generate a swiss roll dataset"
function swiss_roll(n::Int = 1000, noise::Float64=0.05)
    t = (3 * pi / 2) * (1 .+ 2 * rand(n, 1))
    height = 30 * rand(n, 1)
    X = [t .* cos.(t) height t .* sin.(t)] + noise * randn(n, 3)
    labels = vec(rem.(sum([round.(Int, t / 2) round.(Int, height / 12)], dims=2), 2))
    return collect(transpose(X)), labels
end

"Perform spectral decomposition for Ax=λI"
function decompose(M::AbstractMatrix{T}, d::Int) where T<:Real
    W = isa(M, AbstractSparseArray) ? Symmetric(Matrix(M)) : Symmetric(M)
    F = eigen!(W)
    idx = sortperm(F.values)[2:d+1]
    return F.values[idx], F.vectors[:,idx]
end

"Perform spectral decomposition for Ax=λB"
function decompose(A::AbstractMatrix{T}, B::AbstractMatrix{T}, d::Int) where T<:Real
    AA = isa(A, AbstractSparseArray) ? Symmetric(full(A)) : Symmetric(A)
    BB = isa(B, AbstractSparseArray) ? Symmetric(full(B)) : Symmetric(B)
    F = eigen(AA, BB)
    idx = sortperm(F.values)[2:d+1]
    return F.values[idx], F.vectors[:,idx]
end
