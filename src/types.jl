abstract Results

typealias Projection Matrix{Float64}

immutable SpectralResults <: Results
    proj::Projection
    λ::Vector{Float64}
    SpectralResults(proj::Projection, λ::Vector{Float64}) = new(proj, λ)
end

## properties
indim(M::SpectralResults) = size(M.proj, 1)
outdim(M::SpectralResults) = size(M.proj, 2)
projection(M::SpectralResults) = M.proj
eigenvalues(M::SpectralResults) = M.λ

immutable DistSpectralResults <: Results
    res::SpectralResults
    k::Int
    ɛ::Float64
    lcc::Vector{Int}
    DistSpectralResults(proj::Matrix{Float64},
        λ::Vector{Float64}, k::Int) = new(SpectralResults(proj, λ), k)
    DistSpectralResults(proj::Matrix{Float64},
        λ::Vector{Float64}, k::Int, lcc::Vector{Int}) = new(SpectralResults(proj, λ), k, 0.0, lcc)
    DistSpectralResults(proj::Matrix{Float64},
        λ::Vector{Float64}, ɛ::Float64) = new(SpectralResults(proj, λ), 0, ɛ)
    DistSpectralResults(proj::Matrix{Float64},
        λ::Vector{Float64}, ɛ::Float64, lcc::Vector{Int}) = new(SpectralResults(proj, λ), 0, ɛ, lcc)
end

## properties
indim(M::DistSpectralResults) = indim(M.res)
outdim(M::DistSpectralResults) = outdim(M.res)
projection(M::DistSpectralResults) = projection(M.res)
eigenvalues(M::DistSpectralResults) = M.λ
nneighbors(M::DistSpectralResults) = M.k
ccomponent(M::DistSpectralResults) = M.lcc

function Base.show(io::IO, M::SpectralResults)
    println(io, "indim: $(indim(M))")
    println(io, "outdim: $(outdim(M))")
    println(io, "eigenvalues: ")
    Base.showarray(io, M.λ', header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, M.proj, header=false, repr=false)
end

function Base.show(io::IO, M::DistSpectralResults)
    if M.k > 0
        println(io, "nn: $(M.k)")
    end
    if M.ɛ > 0.0
        println(io, "ɛ-radius: $(M.ɛ)")
    end
    # try # Print largest connected component
    #     lcc = M.lcc
    #     println(io, "connected component: ")
    #     Base.showarray(io, lcc', header=false, repr=false)
    #     println(io)
    # end
    show(io, M.res)
end

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