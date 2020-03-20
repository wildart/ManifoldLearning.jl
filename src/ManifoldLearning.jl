module ManifoldLearning

    import Base: show, summary
    import SparseArrays: AbstractSparseArray, SparseMatrixCSC, spzeros, spdiagm, findnz
    import Statistics: mean, std
    import StatsBase: StatsBase, fit, standardize
    import MultivariateStats: outdim, projection,  KernelPCA, transform, transform!,
                              principalvars, dmat2gram, gram2dmat, pairwise
    import LinearAlgebra: eigvals, mul!, svd, qr, Symmetric, eigen, eigen!, diagm, tr, rmul!, I, norm
    import LightGraphs: neighbors, nv, add_edge!, connected_components, vertices,
                        dijkstra_shortest_paths, induced_subgraph, weights
    import SimpleWeightedGraphs: SimpleWeightedGraph

    export

    # Transformation types
    AbstractDimensionalityReduction,
    Isomap,             # Type: Isomap model
    HLLE,               # Type: Hessian Eigenmaps model
    LLE,                # Type: Locally Linear Embedding model
    LTSA,               # Type: Local Tangent Space Alignment model
    LEM,                # Type: Laplacian Eigenmaps model
    DiffMap,            # Type: Diffusion maps model

    ## common interface
    outdim,             # the output dimension of the transformation
    fit,                # perform the manifold learning
    transform,          # the projection matrix
    eigvals,            # eigenvalues from the spectral analysis
    neighbors,          # the number of nearest neighbors used for aproximate local subspace
    vertices,           # vertices of largest connected component
    projection          # the projection matrix (deprecated)

    include("interface.jl")
    include("utils.jl")
    include("nearestneighbors.jl")
    include("isomap.jl")
    include("hlle.jl")
    include("lle.jl")
    include("ltsa.jl")
    include("lem.jl")
    include("diffmaps.jl")

    # deprecated functions
    @deprecate transform(DimensionalityReduction, X; k=k, d=d) fit(DimensionalityReduction, X; k=k, maxoutdim=d)
    @deprecate projection(DimensionalityReductionModel) transform(DimensionalityReductionModel)
end
