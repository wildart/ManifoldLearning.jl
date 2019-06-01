using ManifoldLearning
using Test
import Random

Random.seed!(3483743871)

@testset "ManifoldLearning" begin

    # setup parameters
    k = 12
    d = 2
    X, L = ManifoldLearning.swiss_roll()

    # test NN graph
    D, E = ManifoldLearning.knn(X, k)
	@test size(X,2) == size(D,2) && size(D, 1) == k
	@test size(X,2) == size(E,2) && size(E, 1) == k

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
