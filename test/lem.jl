module TestLEM
    using ManifoldLearning
    using Test

    @testset "TestLEM" begin

	k = 12
	d = 2
	X, L = swiss_roll()
	Y = transform(LEM, X, k=k, d=d)

	@test outdim(Y) == d
	@test size(projection(Y), 2) == size(X, 2)
	@test neighbors(Y) == k
	@test length(eigvals(Y)) == d
	@test length(ccomponent(Y)) > 1
    end
end
