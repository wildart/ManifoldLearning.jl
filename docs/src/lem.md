# Laplacian Eigenmaps

[Laplacian Eigenmaps](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Laplacian_eigenmaps) (LEM) method uses spectral techniques to perform dimensionality reduction. This technique relies on the basic assumption that the data lies in a low-dimensional manifold in a high-dimensional space. The algorithm provides a computationally efficient approach to non-linear dimensionality reduction that has locally preserving properties [^1].

This package defines a [`LEM`](@ref) type to represent a Laplacian Eigenmaps results, and provides a set of methods to access its properties.


```@docs
LEM
fit(::Type{LEM}, X::AbstractArray{T,2}) where {T<:Real}
transform(R::LEM)
```

## References

[^1]: Belkin, M. and Niyogi, P. "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation". Neural Computation, June 2003; 15 (6):1373-1396. DOI:[10.1162/089976603321780317](http://dx.doi.org/doi:10.1162/089976603321780317)
