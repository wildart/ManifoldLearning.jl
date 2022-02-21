using ManifoldLearning
using ManifoldLearning: BruteForce, knn, swiss_roll
using Test
using Statistics
using StableRNGs

rng = StableRNG(83743871)

@testset "Nearest Neighbors" begin
    # setup parameters
    k = 12
    X, _ = swiss_roll(100, rng=rng)
    DD, EE = knn(X,k)
    @test_throws AssertionError knn(zeros(3,10), k)

    NN = fit(BruteForce, X)
    A = ManifoldLearning.adjacency_matrix(NN, X, k)
    @test size(X,2) == size(A,2)
    @test A ≈ ManifoldLearning.adjacency_matrix(collect(eachcol(EE)), collect(eachcol(DD)))

    E, W = ManifoldLearning.adjacency_list(NN, X, k)
    @test size(X,2) == length(W) && length(W[1]) == k
    @test size(X,2) == length(E) && length(E[1]) == k
    @test hcat(E...) == EE
    @test hcat(W...) ≈ DD
    @test A ≈ ManifoldLearning.adjacency_matrix(E, W)

    A = ManifoldLearning.adjacency_matrix(NN, X[:,1:k+1], k)
    @test size(X,2) == size(A,2)
    @test sum(A[k+2:end,k+2:end]) == 0

    A = ManifoldLearning.adjacency_matrix(NN, X, k/7)
    @test size(X,2) == size(A,2)
    @test maximum(A) <= k/7

    E, W = ManifoldLearning.adjacency_list(NN, X, k/7)
    @test size(X,2) == length(W)
    @test size(X,2) == length(E)
    @test maximum(Iterators.flatten(W)) <= k/7

    @test_throws AssertionError ManifoldLearning.adjacency_matrix(NN, X, 101)
    @test_throws AssertionError ManifoldLearning.adjacency_list(NN, X, 101)

    d = zeros(10)
    @test_throws ArgumentError ManifoldLearning.pairwise!(kernel, d, eachcol(X))
    k = (x,y)->x'y
    n = 5
    ManifoldLearning.pairwise!(k, d, eachcol(X[:,1:n]))
    D = ManifoldLearning.pairwise(k, eachcol(X[:,1:n]))
    @testset for i in 1:n, j in i+1:n
        k = n*(i-1) - (i*(i+1))>>1 + j
        @test D[i,j] ≈ d[k]
    end
end

@testset "Laplacian" begin

    A = [0 1 0; 1 0 1; 0 1 0.0]
    L, D = ManifoldLearning.Laplacian(A)
    @test [D[i,i] for i in 1:3] == [1, 2, 1]
    @test D-L == A
    Lsym = ManifoldLearning.normalize!(copy(L), D; α=1/2, norm=:sym)
    @test Lsym ≈ [1 -√.5 0; -√.5 1 -√.5; 0 -√.5 1]
    Lrw = ManifoldLearning.normalize!(copy(L), D; α=1, norm=:rw)
    @test Lrw ≈ [1 -1 0; -0.5 1 -0.5; 0 -1 1]

end

@testset "Manifold Learning" begin
    # setup parameters
    k = 12
    d = 2
    X, L = swiss_roll(;rng=rng)

    # test algorithms
    @testset for algorithm in [Isomap, LEM, LLE, HLLE, LTSA, DiffMap]
        for (k, T) in zip([5, 12], [Float32, Float64])
            # construct KW parameters
            kwargs = [:maxoutdim=>d]
            if algorithm == DiffMap
                push!(kwargs, :t => k)
            else
                push!(kwargs, :k => k)
            end

            # call transformation
            M = fit(algorithm, X; kwargs...)
            Y = predict(M)

            # test results
            @test size(M) == (3, d)
            if k == 5 && (algorithm === LLE || algorithm === LTSA)
                @test size(Y, 2) == 987
            else
                @test size(Y, 2) == size(X, 2)
            end
            @test length(split(sprint(show, M), '\n')) > 1
            @test length(eigvals(M)) == d
            if algorithm !== DiffMap
                @test neighbors(M) == k
                @test length(vertices(M)) > 1
            end
            
            # test if we provide pre-computed Gram matrix
            if algorithm === DiffMap
                kernel = (x, y) -> exp(-sum((x .- y) .^ 2)) # default kernel
                custom_K = ManifoldLearning.pairwise(kernel, eachcol(X), symmetric=true)
                M_custom_K = fit(algorithm, custom_K; kernel=nothing, kwargs...)
                @test isnan(size(M_custom_K)[1])
                @test predict(M_custom_K) ≈ Y
            end
        end
    end
end

@testset "OOS" begin
    n = 200
    k = 5
    d = 10
    ϵ = 0.01

    X, _ = ManifoldLearning.swiss_roll(n ;rng=rng)
    M = fit(Isomap, X; k=k, maxoutdim=d)
    @test all(sum(abs2, predict(M) .- predict(M,X), dims=1) .< eps())

    XX = X + ϵ*randn(rng, size(X))
    @test sqrt(mean((predict(M) - predict(M,XX)).^2)) < 2ϵ
end

