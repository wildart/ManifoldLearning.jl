module ManifoldLearning

    using StatsBase
    using MultivariateStats

    import Base: show, dump, eigvals
    import MultivariateStats: outdim, projection, transform

    export

    ## common
    outdim,             # the output dimension of the transformation
    projection,         # the projection matrix
    eigvals,            # eigenvalues from the spectral analysis
    neighbors,          # the number of nearest neighbors used for building local structure
    ccomponent,         # point indexes of the largest connected component

    transform,          # perform the manifold learning

    # Transformation types
    Isomap,             # Type: Isomap model
    HLLE,               # Type: Hessian Eigenmaps model
    LLE,                # Type: Locally Linear Embedding model
    LTSA,               # Type: Local Tangent Space Alignment model
    LEM,                # Type: Laplacian Eigenmaps model
    DiffMap,            # Type: Diffusion maps model

    # example dataset
    swiss_roll          # swiss roll dataset generator

    include("types.jl")
    include("utils.jl")
    include("isomap.jl")
    include("hlle.jl")
    include("lle.jl")
    include("ltsa.jl")
    include("lem.jl")
    include("diffmaps.jl")
end

