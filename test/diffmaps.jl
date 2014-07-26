module TestDiffMap
	using ManifoldLearning
	using Base.Test

	d = 2
	t = 1
	X, L = swiss_roll()
	Y = transform(DiffMap, X; d=d, t=t)

	@test outdim(Y) == d
	@test size(projection(Y), 2) == size(X, 2)
end

