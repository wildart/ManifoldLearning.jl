using ManifoldLearning
using  Base.Test

X, L = swiss_roll()
I = hlle(X)
@test indim(I) == size(X, 1)-1
@test outdim(I) == size(X, 2)
