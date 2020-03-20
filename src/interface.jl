## Interface

const Projection{T <: Real} = AbstractMatrix{T}

"""
Abstract type for dimensionality reduction methods
"""
abstract type AbstractDimensionalityReduction end

"""
    vertices(R::AbstractDimensionalityReduction)

Returns vertices of largest connected component in the model `R`.
"""
vertices(R::AbstractDimensionalityReduction) = Int[]

"""
    neighbors(R::AbstractDimensionalityReduction)

Returns the number of nearest neighbors used for aproximate local subspace
"""
neighbors(R::AbstractDimensionalityReduction) = 0

"""
    fit(AbstractDimensionalityReduction, X)

Perform model fitting given the data `X`
"""
fit(::Type{AbstractDimensionalityReduction}, X::AbstractMatrix) = throw("Model fitting is not implemented")

"""
    transfrom(R::AbstractDimensionalityReduction)

Returns a reduced space representation of the data given the model `R` inform of  the projection matrix (of size ``(d, n)``), where `d` is a dimension of the reduced space and `n` in the number of the observations. Each column of the projection matrix corresponds to an observation in projected reduced space.
"""
transform(R::AbstractDimensionalityReduction) = throw("Data transformation is not implemented")

"""
    outdim(R::AbstractDimensionalityReduction)

Returns a dimension of the reduced space for the model `R`
"""
outdim(R::AbstractDimensionalityReduction) = 0

"""
    eigvals(R::AbstractDimensionalityReduction)

Returns eignevalues of the reduced space reporesentation for the model `R`
"""
eigvals(R::AbstractDimensionalityReduction) = Float64[]

# Auxiliary functions

show(io::IO, ::MIME"text/plain", R::T) where {T<:AbstractDimensionalityReduction} = summary(io, R)
function show(io::IO, R::T) where {T<:AbstractDimensionalityReduction}
    summary(io, R)
    io = IOContext(io, :limit=>true)
    println(io)
    println(io, "connected component: ")
    Base.show_vector(io, vertices(R))
    println(io)
    println(io, "eigenvalues: ")
    Base.show_vector(io, eigvals(R))
    println(io)
    println(io, "projection:")
    Base.print_matrix(io, transform(R), "[", ",","]")
end
