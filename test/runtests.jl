using ManifoldLearning
import ManifoldLearning: BruteForce, knn, swiss_roll
import StatsBase: pairwise
using Test
import Random

Random.seed!(3483743871)

@testset "Nearest Neighbors" begin

    # setup parameters
    k = 12
    X, L = swiss_roll(100)
    DD, EE = knn(X,k)
    @test_throws AssertionError knn(rand(3,10), k)

    NN = fit(BruteForce, X, k)
    D, E = knn(NN, X)
    @test size(X,2) == size(D,2) && size(D, 1) == k
    @test size(X,2) == size(E,2) && size(E, 1) == k
    @test E == EE
    @test D ≈ DD

    D, E = knn(NN, X[:,1:k+1])
    @test k+1 == size(D,2) && size(D, 1) == k
    @test k+1 == size(E,2) && size(E, 1) == k

    D, E = knn(NN, X[:,1:k+1], self=true)
    @test D[1,:] == zeros(k+1)
    @test E[1,:] == collect(1:k+1)

    @test_throws AssertionError knn(NN, X[:,1:k])
end

@testset "Manifold Learning" begin

    # setup parameters
    k = 12
    d = 2
    X, L = swiss_roll()

    # test algorithms
    #@testset for algorithm in [Isomap, LEM, LLE, HLLE, LTSA, DiffMap]
    @testset for algorithm in [DiffMap]
        for (k, T) in zip([5, 12], [Float32, Float64])
            # construct KW parameters
            kwargs = [:maxoutdim=>d]
            if algorithm == DiffMap
                push!(kwargs, :t => k)
            else
                push!(kwargs, :k => k)
            end

            # call transformation
            Y = fit(algorithm, convert(Array{T}, X); kwargs...)

            # test results
            @test outdim(Y) == d
            @test size(transform(Y), 2) == size(X, 2)
            @test length(split(sprint(show, Y), '\n')) > 1
            @test length(eigvals(Y)) == d
            if algorithm !== DiffMap
                @test neighbors(Y) == k
                @test length(vertices(Y)) > 1
            end
            
            # test if we provide pre-computed Gram matrix
            if algorithm == DiffMap
                kernel = (x, y) -> exp(-sum((x .- y) .^ 2)) # default kernel
                n_obs = size(X)[2]
                #custom_K = pairwise(kernel, eachcol(X), symmetric=true)
                custom_K = zeros(T, n_obs, n_obs)
                for i = 1:n_obs
                    for j = i:n_obs
                        custom_K[i, j] = kernel(convert(Array{T}, X[:, i]), convert(Array{T}, X[:, j]))
                        custom_K[j, i] = custom_K[i, j]
                    end
                end
                Y_custom_K = fit(algorithm, custom_K; kwargs..., kernel=nothing)
                @test Y_custom_K.proj ≈ Y.proj
            end
        end
    end
end

@testset "OOS" begin
    k = 10
    d = 2
    X, _ = swiss_roll()
    M = fit(Isomap, X; k=k, maxoutdim=d)

    @test all(sum(abs2, transform(M) .- transform(M,X), dims=1) .< eps())
end
