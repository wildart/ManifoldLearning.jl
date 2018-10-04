module TestUtils
    using ManifoldLearning
    using Test

    @testset "Utils" begin

	k = 3
	X = rand(3, 15)

	# test k-nn graph
	D, E = ManifoldLearning.find_nn(X, k, excluding=false)
	@test size(X,2) == size(D,2) && size(D, 1)-1 == k
	@test size(X,2) == size(E,2) && size(E, 1)-1 == k

	D, E = ManifoldLearning.find_nn(X, k)
	@test size(X,2) == size(D,2) && size(D, 1) == k
	@test size(X,2) == size(E,2) && size(E, 1) == k


	# test connected components
	CC = ManifoldLearning.components(E)
	@test length(CC) > 0

	# test shortest path
        D = transpose([7 9 14; 7 10 15; 9 10 2; 15 6 0; 6 9 11; 14 2 9])
        E = transpose([2 3 6; 1 3 4; 1 2 6; 2 5 4; 3 4 6; 1 3 5])
	P, PD = ManifoldLearning.dijkstra(D, E, 1)
	@test PD[5] == 20. && PD[6] == 11.
    end
end
