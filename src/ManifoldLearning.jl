module ManifoldLearning

    using LinearAlgebra
    using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, spzeros, spdiagm,
                        findnz, dropzeros!
    using StatsAPI: pairwise
    using Statistics: mean
    using MultivariateStats: NonlinearDimensionalityReduction, KernelPCA,
                             dmat2gram, gram2dmat, transform!, projection
    using Graphs: nv, add_edge!, connected_components, dijkstra_shortest_paths,
                  induced_subgraph, weights, SimpleGraph
    using Random: AbstractRNG, default_rng

    import StatsAPI: fit, predict
    import Base: show, summary, size
    import LinearAlgebra: eigvals
    import Graphs: vertices, neighbors

    export

    # Transformation types
    Isomap,             # Type: Isomap model
    HLLE,               # Type: Hessian Eigenmaps model
    LLE,                # Type: Locally Linear Embedding model
    LTSA,               # Type: Local Tangent Space Alignment model
    LEM,                # Type: Laplacian Eigenmaps model
    DiffMap,            # Type: Diffusion maps model

    ## common interface
    outdim,             # the output dimension of the transformation
    fit,                # perform the manifold learning
    predict,            # transform the data using a given model
    eigvals,            # eigenvalues from the spectral analysis
    neighbors,          # the number of nearest neighbors used for aproximate local subspace
    vertices           # vertices of largest connected component

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
    @deprecate transform(m) predict(m)
    @deprecate transform(m, x) predict(m, x)
    @deprecate outdim(m) size(m)[2]

end
