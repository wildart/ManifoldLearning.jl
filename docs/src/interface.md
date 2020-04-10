# Programming interface

The interface of manifold learning methods in this packages is partially adopted from the packages [StatsBase](https://github.com/JuliaStats/StatsBase.jl), [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) and [LightGraphs.jl](https://github.com/JuliaGraphs/LightGraphs.jl).
You can easily implement additional dimensionality reduction algorithms by implementing the following interface.

## Types and functions

The following functions are currently available from the interface. `AbstractDimensionalityReduction` is a an abstract type required for all implemented algorithms models.

```@docs
AbstractDimensionalityReduction
```

For performing the data dimensionality reduction procedure, a model of the data is constructed by calling [`fit`](@ref) method, and the transformation of the data given the model is done by [`transform`](@ref) method.

```@docs
fit(::Type{AbstractDimensionalityReduction}, X::AbstractMatrix)
transform(R::AbstractDimensionalityReduction)
```

There are auxiliary methods that allow to inspect properties of the constructed model.

```@docs
outdim(R::AbstractDimensionalityReduction)
eigvals(R::AbstractDimensionalityReduction)
vertices(R::AbstractDimensionalityReduction)
neighbors(R::AbstractDimensionalityReduction)
```
