using ManifoldLearning
import ManifoldLearning: BruteForce, knn, swiss_roll
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
            Y = fit(algorithm, convert(Array{T}, X); kwargs...)

            # test results
            @test outdim(Y) == d
            @test size(transform(Y), 2) == size(X, 2)
            @test length(split(sprint(show, Y), '\n')) > 1
            if algorithm !== DiffMap
                @test neighbors(Y) == k
                @test length(eigvals(Y)) == d
            end
            if algorithm === Isomap || algorithm === LEM
                @test length(vertices(Y)) > 1
            end
        end
    end

end
