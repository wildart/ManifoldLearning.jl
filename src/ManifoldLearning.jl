module ManifoldLearning

    import Base: show, dump
    import SparseArrays: AbstractSparseArray, SparseMatrixCSC, spzeros, spdiagm, findnz
    import Statistics: mean, std
    import MultivariateStats: outdim, projection, transform, classical_mds
    import LinearAlgebra: eigvals, mul!, svd, qr, Symmetric, eigen, eigen!, diagm, tr, rmul!, I
    import LightGraphs: neighbors, nv, add_edge!, connected_components, vertices,
                        dijkstra_shortest_paths, induced_subgraph, weights
    import SimpleWeightedGraphs: SimpleWeightedGraph

    export

    ## common
    outdim,             # the output dimension of the transformation
    projection,         # the projection matrix
    eigvals,            # eigenvalues from the spectral analysis
    neighbors,          # the number of nearest neighbors used for building local
    transform,          # perform the manifold learning
    vertices,

    # Transformation types
    Isomap,             # Type: Isomap model
    HLLE,               # Type: Hessian Eigenmaps model
    LLE,                # Type: Locally Linear Embedding model
    LTSA,               # Type: Local Tangent Space Alignment model
    LEM,                # Type: Laplacian Eigenmaps model
    DiffMap             # Type: Diffusion maps model

    abstract type AbstractDimensionalityReduction end
    const Projection{T <: Real} = AbstractMatrix{T}

    include("utils.jl")
    include("transformations.jl")
    include("isomap.jl")
    include("hlle.jl")
    include("lle.jl")
    include("ltsa.jl")
    include("lem.jl")
    include("diffmaps.jl")
end
