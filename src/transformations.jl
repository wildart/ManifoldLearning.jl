### Transformtion
abstract type DataTransform end

# apply the transform
transform!(t::S, x::DenseArray{T,1}) where {T<:AbstractFloat, S<:DataTransform} = transform!(x, t, x)
transform!(t::S, x::DenseArray{T,2}) where {T<:AbstractFloat, S<:DataTransform} = transform!(x, t, x)

transform(t::S, x::DenseArray{T,1}) where {T<:Real, S<:DataTransform} = transform!(Array{T}(undef, size(x)), t, x)
transform(t::S, x::DenseArray{T,2}) where {T<:Real, S<:DataTransform} = transform!(Array{T}(undef, size(x)), t, x)

# reconstruct the original data from transformed values
reconstruct!(t::S, x::DenseArray{T,1}) where {T<:AbstractFloat, S<:DataTransform} = reconstruct!(x, t, x)
reconstruct!(t::S, x::DenseArray{T,2}) where {T<:AbstractFloat, S<:DataTransform} = reconstruct!(x, t, x)

reconstruct(t::S, y::DenseArray{T,1}) where {T<:Real, S<:DataTransform} = reconstruct!(Array{T}(undef, size(y)), t, y)
reconstruct(t::S, y::DenseArray{T,2}) where {T<:Real, S<:DataTransform} = reconstruct!(Array{T}(undef, size(y)), t, y)

# Z-score transformation
struct ZScoreTransform{T<:Real} <: DataTransform
    dim::Int
    mean::AbstractVector{T}
    scale::AbstractVector{T}

    ZScoreTransform{T}() where T = new(0, T[], T[])

    function ZScoreTransform{T}(d::Int, m::AbstractVector{T}, s::AbstractVector{T}) where T
        lenm = length(m)
        lens = length(s)
        lenm == d || lenm == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lens == d || lens == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new(d, m, s)
    end
end

indim(t::ZScoreTransform) = t.dim
outdim(t::ZScoreTransform) = t.dim

# fit a z-score transform
function fit(::Type{ZScoreTransform}, X::DenseArray{T,2}; center::Bool=true, scale::Bool=true) where T<:Real
    d, n = size(X)
    n >= 2 || error("X must contain at least two columns.")

    m = vec(mean(X, dims=2))
    s = std(X, dims=2, mean=m)

    return ZScoreTransform{T}(d, (center ? map(T, vec(m)) : Array{T}([])),
                              (scale ? map(T, vec(s)) : Array{T}([])))
end


function transform!(y::DenseArray{YT,1}, t::ZScoreTransform, x::DenseArray{XT,1}) where {YT<:Real,XT<:Real}
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if x !== y
                copyto!(y, x)
            end
        else
            for i = 1:d
                @inbounds y[i] = x[i] / s[i]
            end
        end
    else
        if isempty(s)
            for i = 1:d
                @inbounds y[i] = x[i] - m[i]
            end
        else
            for i = 1:d
                @inbounds y[i] = (x[i] - m[i]) / s[i]
            end
        end
    end
    return y
end

function transform!(y::DenseArray{YT,2}, t::ZScoreTransform, x::DenseArray{XT,2}) where {YT<:Real,XT<:Real}
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(x,2)
    size(y,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if x !== y
                copyto!(y, x)
            end
        else
            for j = 1:n
                for i = 1:d
                    @inbounds y[i, j] = x[i, j] / s[i]
                end
            end
        end
    else
        if isempty(s)
            for j = 1:n
                for i = 1:d
                    @inbounds y[i,j] = x[i,j] - m[i]
                end
            end
        else
            for j = 1:n
                for i = 1:d
                    @inbounds y[i, j] = (x[i, j] - m[i]) / s[i]
                end
            end
        end
    end
    return y
end

function reconstruct!(x::DenseArray{XT,1}, t::ZScoreTransform, y::DenseArray{YT,1}) where {YT<:Real,XT<:Real}
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if y !== x
                copyto!(x, y)
            end
        else
            for i = 1:d
                @inbounds x[i] = y[i] * s[i]
            end
        end
    else
        if isempty(s)
            for i = 1:d
                @inbounds x[i] = y[i] + m[i]
            end
        else
            for i = 1:d
                @inbounds x[i] = y[i] * s[i] + m[i]
            end
        end
    end
    return x
end

function reconstruct!(x::DenseArray{XT,2}, t::ZScoreTransform, y::DenseArray{YT,2}) where {YT<:Real,XT<:Real}
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(y,2)
    size(x,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    m = t.mean
    s = t.scale

    if isempty(m)
        if isempty(s)
            if y !== x
                copyto!(x, y)
            end
        else
            for j = 1:n
                for i = 1:d
                    @inbounds x[i, j] = y[i, j] * s[i]
                end
            end
        end
    else
        if isempty(s)
            for j = 1:n
                for i = 1:d
                    @inbounds x[i, j] = y[i, j] + m[i]
                end
            end
        else
            for j = 1:n
                for i = 1:d
                    @inbounds x[i, j] = y[i, j] * s[i] + m[i]
                end
            end
        end
    end
    return x
end

# UnitRangeTransform normalization

struct UnitRangeTransform  <: DataTransform
    dim::Int
    unit::Bool
    min::Vector{Float64}
    scale::Vector{Float64}

    UnitRangeTransform() = new(0, false, Float64[], Float64[])

    function UnitRangeTransform(d::Int, unit::Bool, min::Vector{Float64}, max::Vector{Float64})
        lenmin = length(min)
        lenmax = length(max)
        lenmin == d || lenmin == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        lenmax == d || lenmax == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new(d, unit, min, max)
    end
end

indim(t::UnitRangeTransform) = t.dim
outdim(t::UnitRangeTransform) = t.dim

function fit(::Type{UnitRangeTransform}, X::DenseArray{T,2}; unit::Bool=true) where T<:Real
    d, n = size(X)

    tmin = Array{T}(undef, d)
    tmax = Array{T}(undef, d)
    copyto!(tmin, X[:, 1])
    copyto!(tmax, X[:, 1])
    for j = 2:n
        @inbounds for i = 1:d
            if X[i, j] < tmin[i]
                tmin[i] = X[i, j]
            elseif X[i, j] > tmax[i]
                tmax[i] = X[i, j]
            end
        end
    end
    for i = 1:d
        @inbounds tmax[i] = 1.0 / (tmax[i] - tmin[i])
    end
    return UnitRangeTransform(d, unit, tmin, tmax)
end

function transform!(y::DenseArray{YT,1}, t::UnitRangeTransform, x::DenseArray{XT,1}) where {YT<:Real,XT<:Real}
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    tmin = t.min
    tscale = t.scale

    if t.unit
        for i = 1:d
            @inbounds y[i] = (x[i] - tmin[i]) * tscale[i]
        end
    else
        for i = 1:d
            @inbounds y[i] = x[i] * tscale[i]
        end
    end
    return y
end

function transform!(y::DenseArray{YT,2}, t::UnitRangeTransform, x::DenseArray{XT,2}) where {YT<:Real,XT<:Real}
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(x,2)
    size(y,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    tmin = t.min
    tscale = t.scale

    if t.unit
        for j = 1:n
            for i = 1:d
                @inbounds y[i, j] = (x[i,j] - tmin[i]) * tscale[i]
            end
        end
    else
        for j = 1:n
            for i = 1:d
                @inbounds y[i, j] = x[i, j] * tscale[i]
            end
        end
    end
    return y
end

function reconstruct!(x::DenseArray{XT,1}, t::UnitRangeTransform, y::DenseArray{YT,1}) where {YT<:Real,XT<:Real}
    d = t.dim
    length(x) == length(y) == d || throw(DimensionMismatch("Inconsistent dimensions."))

    tmin = t.min
    tscale = t.scale

    if t.unit
        for i = 1:d
            @inbounds x[i] = y[i] / tscale[i] +  tmin[i]
        end
    else
        for i = 1:d
            @inbounds x[i] = y[i] / tscale[i]
        end
    end
    return x
end

function reconstruct!(x::DenseArray{XT,2}, t::UnitRangeTransform, y::DenseArray{YT,2}) where {YT<:Real,XT<:Real}
    d = t.dim
    size(x,1) == size(y,1) == d || throw(DimensionMismatch("Inconsistent dimensions."))
    n = size(y,2)
    size(x,2) == n || throw(DimensionMismatch("Inconsistent dimensions."))

    tmin = t.min
    tscale = t.scale

    if t.unit
        for j = 1:n
            for i = 1:d
                @inbounds x[i,j] = y[i,j] / tscale[i] + tmin[i]
            end
        end
    else
        for j = 1:n
            for i = 1:d
                @inbounds x[i,j] = y[i,j] / tscale[i]
            end
        end
    end
    return x
end
