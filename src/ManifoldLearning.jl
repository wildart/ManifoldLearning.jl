module ManifoldLearning

    export

    ## common
    indim,              # the input dimension of a model
    outdim,             # the output dimension of a model
    projection,         # the projection matrix

    isomap,
    diffusion_maps,
    lem,
    lle,
    hlle,
    ltsa,

    swiss_roll          # swiss roll dataset generator

    include("types.jl")
    include("utils.jl")
    include("isomap.jl")
    include("diffmaps.jl")
    include("lem.jl")
    include("lle.jl")
    include("hlle.jl")
    include("ltsa.jl")
end

