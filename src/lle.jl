# Locally Linear Embedding (LLE)
# ------------------------
# Nonlinear dimensionality reduction by locally linear embedding,
# Roweis, S. & Saul, L., Science 290:2323 (2000)

#### LLE type
struct LLE{T <: Real} <: AbstractDimensionalityReduction
    k::Int
    λ::AbstractVector{T}
    proj::Projection{T}

    LLE{T}(k::Int, λ::AbstractVector{T}, proj::Projection{T})  where T = new(k, λ, proj)
end

## properties
outdim(R::LLE) = size(R.proj, 1)
eigvals(R::LLE) = R.λ
neighbors(R::LLE) = R.k

## show
summary(io::IO, R::LLE) = print(io, "LLE(outdim = $(outdim(R)), neighbors = $(neighbors(R)))")

## interface functions
function fit(::Type{LLE}, X::AbstractMatrix{T};
             maxoutdim::Int=2, k::Int=12,
             tol::Real=1e-5, knn=knn) where {T<:Real}
    # Construct NN graph
    D, E = knn(X, k)
    _, C = largest_component(SimpleWeightedGraph(adjmat(D,E)))
    X = X[:, C]
    n = length(C)

    # Correct indexes of neighbors if more then one connected component
    Ec = E
    if size(E,2) != n
        R = Dict(zip(C, collect(1:n)))
        Ec = zeros(Int,k,n)
        for i in 1 : n
            Ec[:,i] = map(j->get(R,j,C[i]), E[:,C[i]])
        end
    end

    if k > maxoutdim
        @warn("k > maxoutdim: regularization will be used")
    else
        tol = 0
    end

    # Reconstruct weights and compute embedding:
    # M = (I - w)'(I - w) = I - w'I - Iw + w'w
    M = spdiagm(0 => fill(one(T), n))
    Ones = fill(one(T), k, 1)
    for i in 1 : n
        J = Ec[:,i]
        Z = view(X, :, J) .- view(X, :, i)
        G = transpose(Z)*Z
        G += I * tol # regularize
        w = vec(G \ Ones)
        w ./= sum(w)
        ww = w*transpose(w)
        for (l, j) in enumerate(J)
            M[i,j] -= w[l]
            M[j,i] -= w[l]
            for (m, jj) in enumerate(J)
                M[j,jj] = ww[l,m]
            end
        end
    end

    λ, V = decompose(M, maxoutdim)
    return LLE{T}(k, λ, rmul!(transpose(V), sqrt(n)))
end

transform(R::LLE) = R.proj
