# Locally Linear Embedding

[Locally Linear Embedding](http://en.wikipedia.org/wiki/Locally_linear_embedding#Locally-linear_embedding>) (LLE) technique builds a single global coordinate system of lower dimensionality. By exploiting the local symmetries of linear reconstructions, LLE is able to learn the global structure of nonlinear manifolds [^1].

This package defines a [`LLE`](@ref) type to represent a LLE results, and provides a set of methods to access its properties.

```@docs
LLE
fit(::Type{LLE}, X::AbstractArray{T,2}) where {T<:Real}
predict(R::LLE)
```

## References

[^1]: Roweis, S. & Saul, L. "Nonlinear dimensionality reduction by locally linear embedding", Science 290:2323 (2000). DOI:[10.1126/science.290.5500.2323] (http://dx.doi.org/doi:10.1126/science.290.5500.2323)
