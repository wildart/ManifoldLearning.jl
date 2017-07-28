abstract type AbstractResult end

abstract type SpectralResult <: AbstractResult end

const Projection{T <: Real} = AbstractMatrix{T}
