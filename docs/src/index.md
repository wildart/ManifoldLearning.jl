# ManifoldLearning.jl

The package __ManifoldLearning__ aims to provide a library for manifold learning and non-linear dimensionality reduction. It provides set of nonlinear dimensionality reduction methods, such as [`Isomap`](@ref), [`LLE`](@ref), [`LTSA`](@ref), and etc.

## Getting started

To install the package just type

```julia
] add ManifoldLearning
```

A simple example of using the [`Isomap`](@ref) dimensionality reduction method on the build-in Swiss roll dataset, [`ManifoldLearning.swiss_roll`](@ref).

```@repl
using ManifoldLearning
X, _ = ManifoldLearning.swiss_roll();
X
M = fit(Isomap, X)
Y = transform(M)
```

## Methods

| Methods | Description |
|:--------|:------------|
|[`Isomap`](@ref)| Isometric mapping |
|[`LLE`](@ref)| Locally Linear Embedding |
|[`HLLE`](@ref)| Hessian Eigenmaps |
|[`LEM`](@ref)| Laplacian Eigenmaps |
|[`LTSA`](@ref)| Local Tangent Space Alignment |
|[`DiffMap`](@ref)| Diffusion maps |

**Notes:** All methods implemented in this package adopt the column-major convention of JuliaStats: in a data matrix, each column corresponds to a sample/observation, while each row corresponds to a feature (variable or attribute).
