# Programming interface

The interface of manifold learning methods in this packages is partially adopted from the packages [StatsAPI](https://github.com/JuliaStats/StatsAPI.jl),
[MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) and [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl).
You can implement additional dimensionality reduction algorithms by implementing the following interface.

## Dimensionality Reduction

The following functions are currently available from the interface.
`NonlinearDimensionalityReduction` is an abstract type required for all
implemented algorithms models.

```@docs
ManifoldLearning.NonlinearDimensionalityReduction
```

For performing the data dimensionality reduction procedure, a model of the data
is constructed by calling [`fit`](@ref) method, and the transformation of
the data given the model is done by [`predict`](@ref) method.

```@docs
fit(::Type{ManifoldLearning.NonlinearDimensionalityReduction}, X::AbstractMatrix)
predict(R::ManifoldLearning.NonlinearDimensionalityReduction)
```

There are auxiliary methods that allow to inspect properties of the constructed model.

```@docs
size(R::ManifoldLearning.NonlinearDimensionalityReduction)
eigvals(R::ManifoldLearning.NonlinearDimensionalityReduction)
vertices(R::ManifoldLearning.NonlinearDimensionalityReduction)
neighbors(R::ManifoldLearning.NonlinearDimensionalityReduction)
```

## Nearest Neighbors

An additional interface is available for creating an implementation of a nearest
neighbors algorithm, which is commonly used for dimensionality reduction methods.
Use `AbstractNearestNeighbors` abstract type to derive a type for a new
implementation.

```@docs
ManifoldLearning.AbstractNearestNeighbors
```

The above interface requires implementation of the following methods:

```@docs
ManifoldLearning.knn(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer) where T<:Real
ManifoldLearning.inradius(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, r::Real) where T<:Real
```

Following auxiliary methods available for any implementation of
`AbstractNearestNeighbors`-derived type:

```@docs
ManifoldLearning.adjacency_list(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer) where T<:Real
ManifoldLearning.adjacency_matrix(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer) where T<:Real
```

The default implementation uses inefficient ``O(n^2)`` algorithm for nearest
neighbors calculations.

```@docs
ManifoldLearning.BruteForce
```

