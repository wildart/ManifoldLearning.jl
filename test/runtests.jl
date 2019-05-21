using ManifoldLearning
using Test

@testset "ManifoldLearning" begin

    # setup parameters
    k = 12
    d = 2
    X, L = ManifoldLearning.swiss_roll()

    # test NN graph
    D, E = ManifoldLearning.find_nn(X, k)
	@test size(X,2) == size(D,2) && size(D, 1) == k
	@test size(X,2) == size(E,2) && size(E, 1) == k

    # test algorithms
    @testset for algorithm in [Isomap, LEM, LLE, HLLE, LTSA, DiffMap]
        for (k, T) in zip([5, 12], [Float32, Float64])
            # construct KW parameters
            kwargs = [:d=>d]
            if algorithm == DiffMap
                push!(kwargs, :t => k)
            else
                push!(kwargs, :k => k)
            end
            println("$algorithm: $T, $k")

            # call transformation
            Y = transform(algorithm, convert(Array{T}, X); kwargs...)

            # test results
            @test outdim(Y) == d
            @test size(projection(Y), 2) == size(X, 2)
            length(methods(neighbors, (algorithm,))) > 0 && @test neighbors(Y) == k
            length(methods(vertices, (algorithm,))) > 0 && @test length(vertices(Y)) > 1
            length(methods(eigvals, (algorithm,))) > 0 && @test length(eigvals(Y)) == d
        end
    end

end
