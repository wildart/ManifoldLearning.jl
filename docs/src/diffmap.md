# Diffusion maps


[Diffusion maps](http://en.wikipedia.org/wiki/Diffusion_map) leverages the relationship between heat diffusion and a random walk; an analogy is drawn between the diffusion operator on a manifold and a Markov transition matrix operating on functions defined on the graph whose nodes were sampled from the manifold [^1].

This package defines a [`DiffMap`](@ref) type to represent a diffusion map results, and provides a set of methods to access its properties.

```@docs
DiffMap
fit(::Type{DiffMap}, X::AbstractArray{T,2}) where {T<:Real}
transform(R::DiffMap)
ManifoldLearning.kernel(R::DiffMap)
```

## References

[^1]: Coifman, R. & Lafon, S. "Diffusion maps". Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30. DOI:[10.1073/pnas.0500334102](http://dx.doi.org/doi:10.1073/pnas.0500334102)
