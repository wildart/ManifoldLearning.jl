# ManifoldLearning

*A Julia package for manifold learning and nonlinear dimensionality reduction.*

| **Documentation**                                                            | **Build Status**                                                  |
|:----------------------------------------------------------------------------:|:-----------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]    | [![][travis-img]][travis-url] [![][coveralls-img]][coveralls-url] |


## Methods

- Isomap
- Diffusion maps
- Locally Linear Embedding (LLE)
- Hessian Eigenmaps (HLLE)
- Laplacian Eigenmaps (LEM)
- Local tangent space alignment (LTSA)

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add ManifoldLearning
```

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

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://wildart.github.io/ManifoldLearning.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://wildart.github.io/ManifoldLearning.jl/dev

[travis-img]: https://travis-ci.org/wildart/ManifoldLearning.jl.svg?branch=master
[travis-url]: https://travis-ci.org/wildart/ManifoldLearning.jl

[coveralls-img]: https://coveralls.io/repos/github/wildart/ManifoldLearning.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/r/wildart/ManifoldLearning.jl?branch=master
