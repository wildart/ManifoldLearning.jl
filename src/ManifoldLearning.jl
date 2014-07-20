module ManifoldLearning

    import Base: show, dump, eigvals

    export

    ## common
    indim,              # the input dimension of a model
    outdim,             # the output dimension of a model
    projection,         # the projection matrix
    fit,

    # Isomap
    Isomap,             # Type: Isomap model

    eigvals,            # eigenvalues from spectral analysis
    nneighbors,         # the number of nearest neighbors used for building local coordinates
    ccomponent,         # point indexes of the largest connected component

    # HLLE
    HLLE,               # Type: Hessian Eigenmaps model

    # LLE
    LLE,                # Type: Locally Linear Embedding model

    # LTSA
    LTSA,               # Type: Local Tangent Space Alignment model

    # LEM
    LEM,                # Type: Laplacian Eigenmaps model

    # Diffusion maps
    DiffMap,            # Type: Diffusion maps model

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

