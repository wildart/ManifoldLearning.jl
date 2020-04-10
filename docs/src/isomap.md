# Isomap

[Isomap](http://en.wikipedia.org/wiki/Isomap) is a method for low-dimensional embedding. *Isomap* is used for computing a quasi-isometric, low-dimensional embedding of a set of high-dimensional data points[^1].

This package defines a [`Isomap`](@ref) type to represent a Isomap calculation results, and provides a set of methods to access its properties.

```@docs
Isomap
fit(::Type{Isomap}, X::AbstractArray{T,2}) where {T<:Real}
transform(R::Isomap)
transform(R::Isomap, X::Union{AbstractArray{T,1}, AbstractArray{T,2}}) where T<:Real
```

## References

[^1]: Tenenbaum, J. B., de Silva, V. and Langford, J. C. "A Global Geometric Framework for Nonlinear Dimensionality Reduction". Science 290 (5500): 2319-2323, 22 December 2000.
