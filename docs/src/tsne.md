# t-Distributed Stochastic Neighborhood Embedding

The [`t`-Distributed Stochastic Neighborhood Embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) is a statistical dimensionality reduction
methods, based on the original SNE[^1] method with t-distributed variant[^2].
The method constructs a probability distribution over pairwise distances in
the data original space, and then optimizes a similar probability distribution of
the pairwise distances of low-dimensional embedding of the data by minimizing 
the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between two distributions.

This package defines a [`TSNE`](@ref) type to represent a t-SNE model, and provides
a set of methods to access its properties.

```@docs
TSNE
fit(::Type{TSNE}, X::AbstractArray{T,2}) where {T<:Real}
predict(R::TSNE)
```

# References
[^1]: Hinton, G. E., & Roweis, S. (2002). Stochastic neighbor embedding. Advances in neural information processing systems, 15.
[^2]: Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).

