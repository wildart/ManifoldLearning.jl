module TestDiffMap
	using ManifoldLearning
	using Base.Test

	t = 1
	X, L = swiss_roll()
	I = fit(DiffMap, X, t=t)

	@test indim(I) == d
	@test outdim(I) == size(X, 2)
end

