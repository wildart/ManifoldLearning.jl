# t-Distributed Stochastic Neighborhood Embedding (t-SNE)
# -------------------------------------------------------
# Visualizing Data using t-SNE
# L. van der Maaten, G. Hinton, Journal of Machine Learning Research 9 (2008) 2579-2605

"""
    TSNE{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction

The `TSNE` type represents a t-SNE model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct TSNE{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction
    d::Int
    p::Real
    β::AbstractVector{T}
    proj::Projection{T}
    nearestneighbors::NN
end

## properties
size(R::TSNE) = (R.d, size(R.proj, 1))
neighbors(R::TSNE) = R.p

## show
function summary(io::IO, R::TSNE)
    id, od = size(R)
    print(io, "t-SNE{$(R.nearestneighbors)}(indim = $id, outdim = $od, perplexity = $(R.p))")
end

## auxiliary
function perplexities(D::AbstractMatrix{T}, p::Real=30;
                      maxiter::Integer=50, tol::Real=1e-7) where {T<:Real}
    k, n = size(D)
    P = zeros(T, size(D))
    βs = zeros(T, n)
    Ĥ = log(p) # desired entropy
    for (i, Dᵢ) in enumerate(eachcol(D))
        Pᵢ = @view P[:,i]
        β = 1 # precision β = 1/σ²
        βmax, βmin = Inf, 0 
        ΔH = 0
        for j in 1:maxiter
            Pᵢ .= exp.(-β.*Dᵢ)
            div∑Pᵢ = 1/sum(Pᵢ)
            H = -log(div∑Pᵢ) + β*(Dᵢ'Pᵢ)*div∑Pᵢ
            Pᵢ .*= div∑Pᵢ
            ΔH = H - Ĥ
            (abs(ΔH) < tol || β < eps()) && break
            if ΔH > 0
                βmin, β = β, isinf(βmax) ? β*2 : (β + βmax)/2
            else
                βmax, β = β, (β + βmin)/2
            end
        end
        #abs(ΔH) > tol && println("P[$i]: perplexity error is above tolerance: $ΔH")
        βs[i] = β
    end
    P, βs
end

function optimize!(Y::AbstractMatrix{T}, P::AbstractMatrix{T}, r::Integer=1;
                   η::Real=200, exaggeration::Real=12,
                   tol::Real=1e-7, mingain::Real=0.01,
                   maxiter::Integer=100, exploreiter::Integer=250) where {T<:Real}
    m, n = size(Y)
    Q = zeros(T, (n*(n-1))>>1)
    L = zeros(T, n, n)
    ∑Lᵢ = zeros(T, n)
    ∇C = zeros(T, m, n)
    U = zeros(T, m, n)
    G = fill!(similar(Y), 1)

    ∑P = max(sum(P), eps(T))
    P .*= exaggeration/∑P
    α = 0.5 # early exaggeration stage momentum

    minerr = curerr = Inf
    minitr = 0
    for itr in 1:maxiter    
        # Student t-distribution: 1/(1+t²/r)^(r+1)/2
        #@time pairwise!((x,y)->1+sum(abs2, x-y)/r, Q, eachcol(Y), skipdiagonal=true)
        h = 1
        @inbounds for i in 1:n, j in (i+1):n
            s = 0
            @simd for l in 1:m
                s += abs2(Y[l,i]-Y[l,j])
            end
            Q[h] = 1+s/r
            h += 1
        end
        Q .^= -(r+1)/2
        ∑Q = 2*sum(Q)

        # KL[P||Q]
        #@time curerr = 2*(P'*log.(P./Q))

        # ∂C/∂Y = (2(r+1)/r)∑ⱼ (pᵢⱼ - qᵢⱼ)(yᵢ-yⱼ)/(1+||yᵢ-yⱼ||²)ʳ
        #A = unpack((compact(P).-Q./∑Q).*Q, skipdiagonal=true) |> collect
        k = 1
        fill!(∑Lᵢ, 0)
        @inbounds for i in 1:n
            ∑Lⱼ = 0
            for j in (i+1):n
                Qᵢⱼ = Q[k]
                l = (P[i,j] - Qᵢⱼ ./ ∑Q)*Qᵢⱼ
                L[i, j] = l
                ∑Lᵢ[j] += l
                ∑Lⱼ += l
                k +=1
            end
            ∑Lᵢ[i] += ∑Lⱼ
            L[i,i] = -∑Lᵢ[i]
            #ΔY = Y .- view(Y, :, i)
            #∇C[:, i] .= ΔY*A[:, i]
        end
        BLAS.symm!('R', 'U', T(-2(r+1)/r), L, Y, zero(T), ∇C)

        # update embedding
        #@. U = α*U - η*G*∇C
        #@. Y += U
        @inbounds for (y, u, g, c) in zip(eachcol(Y), eachcol(U), eachcol(G), eachcol(∇C))
            @. g = ifelse(u*c>0, max(g*0.8, mingain), g+0.2)
            @. u = α*u - η*g*c
            y .+= u
        end

        # switch off exploration stage
        if exploreiter > 0 && itr >= min(maxiter, exploreiter)
            P .*= 1/exaggeration
            α = 0.8 # late stage momentum
            exploreiter = 0
        end

        # convergence check
        gnorm = norm(∇C)
        gnorm < tol && break

        #println("$itr: ||∇C||=$gnorm, min-gain: $(minimum(G))")
    end    
end

## interface functions
"""
    fit(TSNE, data; p=30, maxoutdim=2, kwargs...)

Fit a t-SNE model to `data`.

# Arguments
* `data`: a matrix of observations. Each column of `data` is an observation.

# Keyword arguments
* `p`: a perplexity parameter (*defaut* `30`).
* `maxoutdim`: a dimension of the reduced space (*defaut* `2`).
* `maxiter`: a total number of iterations for the search algorithm (*defaut* `800`).
* `exploreiter`: a number of iterations for the exploration stage of the search algorithm (*defaut* `200`).
* `tol`: a tolerance threshold (*default* `1e-7`).
* `exaggeration`: a tightness control parameter between the original and the reduced space (*defaut* `12`).
* `initialize`: an initialization parameter for the embedding (*defaut* `:pca`).
* `rng`: a random number generator object for initialization of the initial embedding.
* `nntype`: a nearest neighbor construction class (derived from `AbstractNearestNeighbors`)

# Examples
```julia
M = fit(TSNE, rand(3,100)) # construct t-SNE model
R = predict(M)             # perform dimensionality reduction
```
"""
function fit(::Type{TSNE}, X::AbstractMatrix{T}; p::Real=30, maxoutdim::Integer=2,
             maxiter::Integer=800, exploreiter::Integer=200,
             exaggeration::Real=12, tol::Real=1e-7, initialize::Symbol=:pca,
             rng::AbstractRNG=default_rng(), nntype=BruteForce) where {T<:Real}

    d, n = size(X)
    k = min(n-1, round(Int, 3p))
    # Construct NN graph
    NN = fit(nntype, X)
    
    # form distance matrix
    D = adjacency_matrix(NN, X, k, symmetric=false)
    D .^= 2 # sq. dists
    I, J, V = findnz(D)
    Ds = reshape(V, k, :)

    # calculate perplexities & corresponding conditional probabilities matrix P
    Px, βs = perplexities(Ds, p, tol=tol)
    P = sparse(I, J, reshape(Px,:))
    P .+= P' # symmetrize
    P ./= max(sum(P), eps(T))

    # form initial embedding and optimize it
    Y = if initialize == :pca
        predict(fit(PCA, X, maxoutdim=maxoutdim), X)
    elseif initialize == :random
        randn(rng, T, maxoutdim, n).*T(1e-4)
    else
        error("Uknown initialization method: $initialize")
    end
    dof = max(maxoutdim-1, 1)
    optimize!(Y, P, dof; maxiter=maxiter, exploreiter=exploreiter, tol=tol,
              exaggeration=exaggeration, η=max(n/exaggeration/4, 50))

    return TSNE{nntype, T}(d, p, βs, Y, NN) 
end
    
"""
    predict(R::TSNE)

Transforms the data fitted to the t-SNE model `R` into a reduced space representation.
"""
predict(R::TSNE) = R.proj

