module TestLEM
	using ManifoldLearning
	using Base.Test

	k = 12
	d = 2
	X, L = swiss_roll()
	I = fit(LEM, X, k=k, d=d)

	@test indim(I) == d
	@test outdim(I) == size(X, 2)
	@test nneighbors(I) == k
	@test length(eigvals(I)) == d
	@test length(ccomponent(I)) > 1
end
