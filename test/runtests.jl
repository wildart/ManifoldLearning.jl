using ManifoldLearning
using Test

include("utils.jl")
include("transformations.jl")

@testset "ManifoldLearning" begin

    # setup parameters
    k = 12
    d = 2
    t = 1
    X, L = ManifoldLearning.swiss_roll()

    @testset for algorithm in [Isomap, HLLE, LLE, LEM, LTSA, DiffMap]

        # construct KW parameters
        kwargs = [:d=>d]
        if algorithm == DiffMap
            push!(kwargs, :t => t)
        else
            push!(kwargs, :k => k)
        end

        # call transformation
        Y = transform(algorithm, X; kwargs...)

        # test results
        @test outdim(Y) == d
        @test size(projection(Y), 2) == size(X, 2)
        length(methods(neighbors, (algorithm,))) > 0 && @test neighbors(Y) == k
        length(methods(ccomponent, (algorithm,))) > 0 && @test length(ccomponent(Y)) > 1
        length(methods(eigvals, (algorithm,))) > 0 && @test length(eigvals(Y)) == d
    end

end
