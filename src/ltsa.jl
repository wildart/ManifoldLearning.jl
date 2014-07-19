# Local tangent space alignment (LTSA)
# ---------------------------
# Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment,
# Zhang, Zhenyue; Hongyuan Zha (2004),  SIAM Journal on Scientific Computing 26 (1): 313–338.
# doi:10.1137/s1064827502419154.

typealias LTSA DistSpectralResults

function ltsa(X::Matrix; d::Int=2, k::Int=12)
    n = size(X, 2)

    # Construct NN graph
    D, I = find_nn(X, k)

    B = spzeros(n,n)
    for i=1:n
        # re-center points in neighborhood
        μ = mean(X[:,I[:,i]],2)
        δ_x = X[:,I[:,i]] .- μ

        # Compute orthogonal basis H of θ'
        θ_t = svdfact(δ_x)[:V][:,1:d]
        #θ_t = (svdfact(δ_x)[:U][:,1:d]'*δ_x)'
        H = full(qrfact(θ_t)[:Q])

        # Construct alignment matrix
        G = hcat(ones(k)./sqrt(k), H)
        B[I[:,i], I[:,i]] =  eye(k) - G*G'
    end

    # Align global coordinates
    # λ, E = eigs(B, nev=d+1, which="SA", tol=0.0)
    # λi = sortperm(λ)[2:d+1]
    # return LTSA(d, k, λ[λi], E[:, λi]')

    F = eigfact!(Symmetric(full(B)))
    λi = sortperm(F[:values])[2:d+1]
    return LTSA(F[:vectors][:, λi]', F[:values][λi], k)
end
