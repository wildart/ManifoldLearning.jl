module TestIsomap
    using ManifoldLearning
    using Test

    @testset "Isomap" begin

	k = 12
	d = 2
	X, L = swiss_roll()
	Y = transform(Isomap, X, k=k, d=d)

	@test outdim(Y) == d
	@test size(projection(Y), 2) == size(X, 2)
	@test neighbors(Y) == k
	@test length(ccomponent(Y)) > 1
    end
end
