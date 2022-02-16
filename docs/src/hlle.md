# Hessian Eigenmaps

The Hessian Eigenmaps (Hessian LLE, HLLE) method adapts the weights in [`LLE`](@ref) to minimize the [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) operator. Like [`LLE`](@ref), it requires careful setting of the nearest neighbor parameter. The main advantage of Hessian LLE is the only method designed for non-convex data sets [^1].

This package defines a [`HLLE`](@ref) type to represent a Hessian LLE results, and provides a set of methods to access its properties.

```@docs
HLLE
fit(::Type{HLLE}, X::AbstractArray{T,2}) where {T<:Real}
predict(R::HLLE)
```

# References
[^1]: Donoho, D. and Grimes, C. "Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data", Proc. Natl. Acad. Sci. USA. 2003 May 13; 100(10): 5591–5596. DOI:[10.1073/pnas.1031596100](http://dx.doi.org/doi:10.1073/pnas.1031596100)
