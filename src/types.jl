abstract AbstractResult

abstract SpectralResult <: AbstractResult

typealias Projection Matrix{Float64}

type Diffmap
    d::Int
    t::Int
    K::Matrix{Float64}
    Y::Matrix{Float64}
end

function Base.show(io::IO, res::Diffmap)
    println(io, "Dimensionality:")
    show(io, res.d)
    print(io, "\n\n")
    println(io, "Timesteps:")
    show(io, res.t)
    print(io, "\n\n")
    println(io, "Kernel:")
    show(io, res.K)
    print(io, "\n\n")
    println(io, "Embedding:")
    show(io, res.Y)
end