abstract AbstractResult

abstract SpectralResult <: AbstractResult

typealias Projection{T <: Real} AbstractMatrix{T}

type Diffmap{T <: Real}
    d::Int
    t::Int
    K::AbstractMatrix{T}
    Y::AbstractMatrix{T}
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