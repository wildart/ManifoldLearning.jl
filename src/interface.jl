## Interface

const Projection{T <: Real} = AbstractMatrix{T}

"""
    vertices(R::NonlinearDimensionalityReduction)

Returns vertices of largest connected component in the model `R`.
"""
vertices(R::NonlinearDimensionalityReduction) = Int[]

"""
    neighbors(R::NonlinearDimensionalityReduction)

Returns the number of nearest neighbors used for aproximate local subspace
"""
neighbors(R::NonlinearDimensionalityReduction) = 0

"""
    fit(NonlinearDimensionalityReduction, X)

Perform model fitting given the data `X`
"""
fit(::Type{NonlinearDimensionalityReduction}, X::AbstractMatrix; kwargs...) = throw("Model fitting is not implemented")

"""
    predict(R::NonlinearDimensionalityReduction)

Returns a reduced space representation of the data given the model `R` inform of  the projection matrix (of size ``(d, n)``), where `d` is a dimension of the reduced space and `n` in the number of the observations. Each column of the projection matrix corresponds to an observation in projected reduced space.
"""
predict(R::NonlinearDimensionalityReduction) = throw("Data transformation is not implemented")

"""
    size(R::NonlinearDimensionalityReduction)

Returns a tuple of the input and reduced space dimensions for the model `R`
"""
size(R::NonlinearDimensionalityReduction) = (0,0)

"""
    eigvals(R::NonlinearDimensionalityReduction)

Returns eignevalues of the reduced space reporesentation for the model `R`
"""
eigvals(R::NonlinearDimensionalityReduction) = Float64[]

# Auxiliary functions

show(io::IO, ::MIME"text/plain", R::T) where {T<:NonlinearDimensionalityReduction} = summary(io, R)
function show(io::IO, R::T) where {T<:NonlinearDimensionalityReduction}
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
