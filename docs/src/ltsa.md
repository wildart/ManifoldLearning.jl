# Local Tangent Space Alignment

[Local tangent space alignment](http://en.wikipedia.org/wiki/Local_tangent_space_alignment) (LTSA) is a method for manifold learning, which can efficiently learn a nonlinear embedding into low-dimensional coordinates from high-dimensional data, and can also reconstruct high-dimensional coordinates from embedding coordinates [^1].

This package defines a [`LTSA`](@ref) type to represent a local tangent space alignment results, and provides a set of methods to access its properties.

```@docs
LTSA
fit(::Type{LTSA}, X::AbstractArray{T,2}) where {T<:Real}
transform(R::LTSA)
```

## References
[^1]: Zhang, Zhenyue; Hongyuan Zha. "Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment". SIAM Journal on Scientific Computing 26 (1): 313–338, 2004. DOI:[10.1137/s1064827502419154](http://dx.doi.org/doi:10.1137/s1064827502419154)
