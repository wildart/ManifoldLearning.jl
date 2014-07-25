module TestDiffMap
	using ManifoldLearning
	using Base.Test

	t = 1
	X, L = swiss_roll()
	Y = transform(DiffMap, X, t=t)

	@test outdim(Y) == d
	@test size(projection(Y), 2) == size(X, 2)
end

