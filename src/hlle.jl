# Hessian Eigenmaps (HLLE)
# ---------------------------
# Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data,
# D. Donoho and C. Grimes, Proc Natl Acad Sci U S A. 2003 May 13; 100(10): 5591–5596

#### HLLE type
immutable HLLE <: SpectralResult
    k::Int
    λ::Vector{Float64}
    proj::Projection

    HLLE(k::Int, λ::Vector{Float64}, proj::Projection) = new(k, λ, proj)
end

## properties
indim(M::HLLE) = size(M.proj, 1)
outdim(M::HLLE) = size(M.proj, 2)
projection(M::HLLE) = M.proj

eigvals(M::HLLE) = M.λ
nneighbors(M::HLLE) = M.k

## show & dump
function show(io::IO, M::HLLE)
    print(io, "Hessian Eigenmaps(indim = $(indim(M)), outdim = $(outdim(M)), nneighbors = $(nneighbors(M)))")
end

function dump(io::IO, M::HLLE)
    show(io, M)
    println(io, "eigenvalues: ")
    Base.showarray(io, M.λ', header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

## interface functions
function fit(::Type{HLLE}, X::DenseMatrix{Float64}; d::Int=2, k::Int=12)
    n = size(X, 2)

    # Identify neighbors
    D, I = find_nn(X, k)

    # Obtain tangent coordinates and develop Hessian estimator
    hs = int(d*(d+1)/2)
    W = spzeros(hs*n,n)
    for i=1:n
        # re-center points in neighborhood
        μ = mean(X[:,I[:,i]],2)
        N = X[:,I[:,i]] .- μ
        # calculate tangent coordinates
        #tc = svdfact(N')[:U][:,1:d]
        tc = svdfact(N)[:V][:,1:d]

        # Develop Hessian estimator
        Yi = [ones(k) tc zeros(k,hs)]
        for ii=1:d
            Yi[:,d+ii+1] = tc[:,ii].^2
        end
        yi = 2(1+d)
        for (ii,jj) in combinations(1:d,2)
            Yi[:, yi] = tc[:, ii] .* tc[:, jj]
            yi += 1
        end
        F = qrfact(Yi)
        H = full(F[:Q])[:,d+2:end]'
        W[(i-1)*hs+(1:hs),I[:,i]] = H
    end

    # decomposition
    λ, V = decompose(W'*W, d)
    return HLLE(k, λ, V' .* sqrt(n))
end