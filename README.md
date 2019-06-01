# ManifoldLearning [![Build Status](https://travis-ci.org/wildart/ManifoldLearning.jl.svg?branch=master)](https://travis-ci.org/wildart/ManifoldLearning.jl) [![Coverage Status](https://coveralls.io/repos/wildart/ManifoldLearning.jl/badge.png?branch=master)](https://coveralls.io/r/wildart/ManifoldLearning.jl?branch=master)

A Julia package for manifold learning and nonlinear dimensionality reduction.

## Methods

- Isomap
- Diffusion maps
- Locally Linear Embedding (LLE)
- Hessian Eigenmaps (HLLE)
- Laplacian Eigenmaps (LEM)
- Local tangent space alignment (LTSA)

## Examples

A simple example of using the *Isomap* reduction method.

```julia
julia> X, _ = ManifoldLearning.swiss_roll();

julia> X
3×1000 Array{Float64,2}:
  -3.19512  3.51939   -0.0390153  …  -9.46166   3.44159
  29.1222   9.99283    2.25296       25.1417   28.8007
 -10.1861   6.59074  -11.037         -1.04484  13.4034

julia> M = fit(Isomap, X)
Isomap(outdim = 2, neighbors = 12)

julia> Y = transform(M)
2×1000 Array{Float64,2}:
 11.0033  -13.069   16.7116  …  -3.26095   25.7771
 18.4133   -6.2693  10.6698     20.0646   -24.8973
```

## Performance

Most of the methods use *k*-nearest neighbors method for constructing local subspace representation. By default, neighbors are computed from a *distance matrix* of a dataset. This is not an efficient method, especially, for large datasets.

Consider using a custom *k*-nearest neighbors function, e.g. from [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) or [FLANN.jl](https://github.com/wildart/FLANN.jl).

See example of custom `knn` function [here](misc/nearestneighbors.jl).

## Resources

- **Documentation:** <http://manifoldlearningjl.readthedocs.org/en/latest/index.html>

