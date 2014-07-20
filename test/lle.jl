module TestLLE
	using ManifoldLearning
	using Base.Test

	k = 12
	d = 2
	X, L = swiss_roll()
	I = fit(LLE, X, k=k, d=d)

	@test indim(I) == d
	@test outdim(I) == size(X, 2)
	@test nneighbors(I) == k
	@test length(eigvals(I)) == d
end

