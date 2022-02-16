# ManifoldLearning.jl

The package __ManifoldLearning__ aims to provide a library for manifold learning
and non-linear dimensionality reduction. It provides set of nonlinear dimensionality
reduction methods, such as [`Isomap`](@ref), [`LLE`](@ref), [`LTSA`](@ref), and etc.

## Getting started

To install the package just type

```julia
] add ManifoldLearning
```

```@setup EG
using Plots
gr(fmt=:svg)
```

The following example shows how to apply [`Isomap`](@ref) dimensionality reduction method
to the build-in S curve dataset.

```@example EG
using ManifoldLearning
X, L = ManifoldLearning.scurve(segments=5);
scatter3d(X[1,:], X[2,:], X[3,:], c=L,palette=cgrad(:default),ms=2.5,leg=:none,camera=(10,10))
```

Now, we perform dimensionality reduction procedure and plot the resulting dataset:

```@example EG
Y = predict(fit(Isomap, X))
scatter(Y[1,:], Y[2,:], c=L, palette=cgrad(:default), ms=2.5, leg=:none)
```

Following dimensionality reduction methods are implemented in this package:

| Methods | Description |
|:--------|:------------|
|[`Isomap`](@ref)| Isometric mapping |
|[`LLE`](@ref)| Locally Linear Embedding |
|[`HLLE`](@ref)| Hessian Eigenmaps |
|[`LEM`](@ref)| Laplacian Eigenmaps |
|[`LTSA`](@ref)| Local Tangent Space Alignment |
|[`DiffMap`](@ref)| Diffusion maps |

**Notes:** All methods implemented in this package adopt the column-major convention of JuliaStats: in a data matrix, each column corresponds to a sample/observation, while each row corresponds to a feature (variable or attribute).
