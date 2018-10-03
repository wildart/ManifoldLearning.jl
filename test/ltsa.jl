module TestLTSA
    using ManifoldLearning
    using Test

    @testset "LTSA" begin

	k = 12
	d = 2
	X, L = swiss_roll()
	Y = transform(LTSA, X; k=k, d=d)

	@test outdim(Y) == d
	@test size(projection(Y), 2) == size(X, 2)
	@test neighbors(Y) == k
	@test length(eigvals(Y)) == d
    end
end
