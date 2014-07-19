using ManifoldLearning
using Base.Test

X, L = swiss_roll(2000)
I = ltsa(X)
@test indim(I) == size(X, 1)-1
@test outdim(I) == size(X, 2)