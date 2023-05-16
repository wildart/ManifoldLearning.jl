# TriMap 
# -------------------------------------------------------
# TriMap: Large-scale Dimensionality Reduction Using Triplets,
# Ehsan Amid, Manfred K. Warmuth, 2022, arXiv:1910.00204

"""
TriMap{NN <: AbstractNearestNeighbors, T <: Real} <:NonlinearDimensionalityReduction

The `TriMap` type represents a TriMap model constructed for `T` type data with a help of the `NN` nearest neighbor algorithm.
"""
struct TriMap{NN <: AbstractNearestNeighbors, T<: Real} <: NonlinearDimensionalityReduction
    d :: Int
    m₁ :: Int
    m₂ :: Int 
    r :: Int
    nearestneighbor :: NN
    proj :: Projection{T} 
end

## Fit and predict functions
fit(::Type{TriMap}, 
    X::AbstractMatrix{T}, 
    maxoutdim :: Integer=2, 
    maxiter :: Integer=400, 
    initialize:: Symbol=:pca, 
    lr :: T = 0.5,
    weight_temp :: T = 0.5, 
    m₁ :: Int = 10, 
    m₂ :: Int = 5, 
    r :: Int = 5, 
    switch_iter = 250,
    final_momentum = 0.8, 
    init_momentum = 0.5, 
    increase_gain = 0.2, 
    damp_gain = 0.8, 
    min_gain = 0.01, 
    nntype = BruteForce) where {T<:Real} = trimap(X, maxoutdim, maxiter, 
                initialize, lr, weight_temp, m₁, m₂, 
                r, switch_iter, final_momentum, 
                init_momentum, increase_gain, damp_gain, 
                min_gain, nntype)

predict(T :: TriMap) = T.proj

## properties
size(R::TriMap) = (R.d, size(R.proj, 1))
neighbors(R::TriMap) = R.m₁
sampledTriplets(R:: TriMap) = R.m₁ * R.m₂ + R.r

## show
function summary(io::IO, R::TriMap)
    id, od = size(R)
    print(io, "TriMap{$(R.nearestneighbor)}(indim = $id, outdim = $od)")
end



# Implementation of the tempered log_t with temperature t and data x 
# (x^{1-t} - 1)/(1-t).
@inline function tempered_log(x :: AbstractVector{T}, t :: T) where {T <: Real}
    if abs(t - 1.0) < 1e-5
        return log.(x)
    else
        return @. (x^(1.0-t) - 1.0) / (1.0 - t)
    end
end


# Implements the squared euclidean distance. Note that this is 
# the distance for matrices, i.e. each point is a column and thus 
# we sum over the rows and obtain a vector of distances for each
# column.
@inline function squared_euclidean_distance(x1 :: AbstractMatrix{T}, 
                        x2 :: AbstractMatrix{T}) where {T <: Real}
    sum(x-> x.^2, x1 .- x2, dims=1) 
end

#   Implements the squared euclidean distance but between two vectors. 
#   This is basically the mathematical definition of  |x - y|^2_2 for 
#   two R^d vectors x,y.
@inline function squared_euclidean_distance(x1 :: AbstractVector{T}, x2 :: AbstractVector{T}) where {T <: Real}
    sum(x -> x.^2, (x1-x2))
end

# Implements the distance between two vectors' coordinate but don't compute 
# the norm (i.e. we don't sum over the coordinates).
function squared_euclidean_distance_cmp(x1 :: AbstractVector{T}, x2 :: AbstractVector{T}) where {T <: Real}
    @. (x1-x2)^2    
end


#    Implements the derivative of the euclidean distance based on the argument `i`, 
#    for i = 1, it corresponds to the partial derivative wrt y, for i = 2, we have 
#    the derivative wrt ̂y.
function squared_euclidean_dst_deriv(i :: Int, 
        y :: AbstractVector{T}, 
        x :: AbstractVector{T}) where {T<: Real}
    return 2 .* (y .- x) .* ifelse(i % 2 == 0, 1, -1) 
end


# Generate a sample of size `n_sample` in a range `[1, max_int]`
# such that each element of the sample doesn't belong to `rejects`.
#
# i.e. If n_sample = 3, max_int = 10, rejects = [1,2,3,4,5]
#     then a possible outcome are 
#     [6,7,8], [6,6,6], [6,7,7], ..., [9,9,10], ...
@inline function rejection_sample(n_sample :: Int, max_int :: Int, rejects :: AbstractArray{Int}) 
    result = Array{Int,1}(undef, n_sample) 
    rr = 1:max_int
    i  = 1
    @inbounds while i <= n_sample
        j = rand(rr)
        if !(j in rejects)
            result[i] = j
            i += 1
        end
    end

    return result
end


# This does the same as rejection sampling except it rejects also the 
# points on the same column of the results in order to not have a triples of (i, j, j).
@inline function rejection_sample_without_repetitions(shape :: Pair{Int, Int}, 
    max_int :: Int, rejects :: AbstractArray{Int})
    rows, col = shape 
    result = Array{Int, 2}(undef, rows, col)
    for i in 1:col
        rejection_array = rejects |> copy
        for j in 1:rows
            v = rejection_sample(1, max_int, rejection_array) 
            result[j,i] = v[1]
            push!(rejection_array, v[1])
        end
    end

    return result
end

# Generate r random points, if the j points are farther than the k points, 
# then switch the indexes.
@inline function generate_random_triplets(points :: AbstractMatrix{T}, 
                                    n_points :: Int,
                                    i :: Int, 
                                    r :: Int) where {T <: Real}
    random_indexes = rejection_sample_without_repetitions(2 => r, n_points, [i])
    ppp     = repeat([points[i]], r)
    dsts1   = squared_euclidean_distance_cmp(ppp, points[random_indexes[1,:]])
    dsts2   = squared_euclidean_distance_cmp(ppp, points[random_indexes[2,:]])
    # check what distances have to be switched
    change  = dsts1 .> dsts2 
    # swap some parts of the rows if the distances between d_{ij} > d_{ik}.
    random_indexes[:,change] = random_indexes[[2,1], change]

    return random_indexes
end

# Generate a collection of triplets, m₁ are the number of points in the 
# neighborhood for each point, m₂ the value of non-neighbors sampled 
# at random, r is the number of random triplets associated to a given 
# point. Moreover, the function computes the weight associated to the 
# loss function.
@inline function generate_triplets(points :: AbstractMatrix{T}, 
                nn :: AbstractNearestNeighbors,
                m₁ :: Int = 10, # Number of k-neighbors
                m₂ :: Int = 5,  # Number of non-neighbors sampled at random
                r :: Int = 5,
                weight_temp :: T = 0.5) where {T <: Real}
    d, n_points = points |> size
    idx, dsts = knn(nn, points, m₁)
    
    # the nearest neighbors for each point i is given by the 
    # idx[i] (which returns indexes!), so we only have to construct 
    # the triplets in a nice way 
    mm = m₁ * m₂
    mmr = m₁ * m₂ + r
    triplets = zeros(Int, 3, n_points * mmr)
    @inbounds triplets[1, :] = repeat(1:n_points, inner = mmr)

    @inbounds for i in 0:(n_points-1)
        # neighbors points
        neighbors = idx[i+1]
        @inline triplets[2, (i * mmr + 1):(i * mmr + mm)] = repeat(neighbors, inner=m₂)
        # out points 
        out_points = rejection_sample(mm, n_points, vcat(i, neighbors))
        @inline triplets[3, (i * mmr + 1):(i * mmr + mm)] = out_points
        # Sample every node except of the node i, because the node i is already
        # at the ''top row'' 
        random_points = generate_random_triplets(points, n_points, i+1, r) 
        triplets[2:3, (i*mmr + mm + 1) : (i*mmr + mm + r)] = random_points
    end

    ### Here we start generating the weights as a tempered log of the 
    ### sum of 1 + tilde{w}_{ijk} - w_min, note that this weights are 
    ### depending only on the initial embedding of the data and not 
    ### on the new dimensionally reduced embedding.
    pointsI = points[:, triplets[1,:]]
    pointsJ = points[:, triplets[2,:]]
    pointsK = points[:, triplets[3,:]]

    # compute significance to enchance the analysis by density of close elements
    # OLD CODE: sig = [sqrt(sum(dst[4:6])/3) for dst in dsts]
    sig = map(dd -> sum(x->/(x,3), dd[4:6]), dsts)
    sigI = sig[triplets[1,:]]
    sigJ = sig[triplets[2,:]]
    sigK = sig[triplets[3,:]]

    # distances normalized by the significance 
    dij = vec(squared_euclidean_distance(pointsI, pointsJ)) ./ (sigI .* sigJ)
    dik = vec(squared_euclidean_distance(pointsI, pointsK)) ./ (sigI .* sigK)

    # take the tempered log of the distances to compute the weights.
    weights = dik - dij
    weights .-= minimum(weights)
    weights = tempered_log(1 .+ weights, weight_temp)

    return triplets, weights
end

# Computes the tripmap metric or loss function.
@inline function trimap_metric(embedding :: AbstractMatrix{T}, 
    triplets :: AbstractMatrix{Int},
    weights :: AbstractVector{T}) where {T <: Real}

    d, n = size(triplets)
    # evaluate the distances
    @inbounds sim_distances = 1.0 .+ squared_euclidean_distance(embedding[:, triplets[1, :]], embedding[:, triplets[2, :]])
    @inbounds out_distances = 1.0 .+ squared_euclidean_distance(embedding[:, triplets[1, :]], embedding[:, triplets[3, :]])

    return @. $(sum)(weights / (1.0 + out_distances / sim_distances)) / n
end

# Compute the gradient for the trimap loss function in case of a single point / triple.   
@inline function local_trimap_grad_i(i :: Int, triple_of_points :: AbstractMatrix{T}) where {T <: Real}
    dij = squared_euclidean_distance(triple_of_points[:,1], triple_of_points[:,2])
    dik = squared_euclidean_distance(triple_of_points[:,1], triple_of_points[:,3])

    den = (dij + dik + 2.0)^2 
    
    # l_{ijk} derived in y_i
    if i == 1
        ddij = squared_euclidean_dst_deriv(1, triple_of_points[:,1], triple_of_points[:,2])
        ddik = squared_euclidean_dst_deriv(1, triple_of_points[:,1], triple_of_points[:,3])
        return @. ((1+dij) * ddik - (1+dik) * ddij) / den
    # l_{ijk} derived in y_j
    elseif i == 2
        ddij = squared_euclidean_dst_deriv(2, triple_of_points[:,1], triple_of_points[:,2])
        return @. -((1+dik) * ddij / den)
    # l_{ijk} derived in y_k
    elseif i == 3
        ddik = squared_euclidean_dst_deriv(2, triple_of_points[:,1], triple_of_points[:,3])
        return @. (1+dij) * ddik / den
    else
        error("This can't happen, triplets consists of 3 elements only!")
    end
end


# TriMap gradient of the loss function 
@inline function trimap_loss_grad!(grad :: AbstractMatrix{T}, 
    embedding :: AbstractMatrix{T},
    triplets :: AbstractMatrix{Int},
    weights :: AbstractVector{T}) where {T <: Real}
    
    
    d, n_points = size(embedding)
    grad .= 0
    r, n_triplets = size(triplets)
    @simd for t in 1:n_triplets 
        @inbounds triple_points = embedding[:,triplets[:,t]]
        @simd for i in 1:r
            gradi = local_trimap_grad_i(i, triple_points)
            @inbounds grad[:,triplets[i,t]] .+= gradi .* weights[t]
        end
    end
        
end


# Optimizer for the embedding optimization using delta-bar-delta
# method.
#
# grad: gradient
# gain: parameter for the optimization 
# vel: velocity/speed for the optimization
# lr : learning rate
# itr: iteration number
@inline function update_embedding!(embedding, grad, vel, gain, lr, itr, 
    switch_iter = 250, final_momentum = 0.8, init_momentum = 0.5, 
    increase_gain = 0.2, damp_gain = 0.8, min_gain = 0.01)
    gamma = ifelse(itr > switch_iter, final_momentum, init_momentum)

    # optimizer via delta-bar-delta method
    check           = @. sign(vel) != sign(grad)
    @. gain[check]  = gain[check] + increase_gain
    @. gain[~check] = max(gain[~check] * damp_gain, min_gain)
    @. vel          = gamma * vel - lr * gain * grad 
    @. embedding    = embedding + vel 
end

# Implements the routine as described in 
# https://arxiv.org/abs/1910.00204
@inline function trimap(X :: AbstractMatrix{T}, 
                maxoutdim :: Integer=2, 
                maxiter :: Integer=400, 
                initialize:: Symbol=:pca,
                lr :: T = 0.5,
                weight_temp :: T = 0.5,
                m₁ :: Int = 10,
                m₂ :: Int = 5,
                r :: Int = 5,
                switch_iter :: Int = 250,
                final_momentum :: T = 0.8, 
                init_momentum :: T = 0.5, 
                increase_gain :: T = 0.2, 
                damp_gain :: T = 0.8, 
                min_gain :: T = 0.01, 
                nntype = BruteForce) where {T <: Real}
    
    d, n = size(X)

    Y = if initialize == :pca 
            predict(fit(PCA, X, maxoutdim=maxoutdim), X)
        elseif initialize == :random
            rand(T, maxoutdim, n)
        else error("Unknown initialization")
    end

    # Neareest neighbors
    NN = fit(nntype, X) 
    
    # initialize triplets and weights 
    triplets, weights = generate_triplets(X, NN, m₁, m₂, r, weight_temp)

    # Optimization of the embedding
    gain = zeros(T, size(Y))
    vel  = zeros(T, size(Y))
    grad = zeros(T, size(Y))
    @inbounds for i in 1:maxiter
        gamma = ifelse(i > switch_iter, final_momentum, init_momentum)
        trimap_loss_grad!(grad, Y .+ gamma .* vel, triplets, weights)
        # NOTE: This is the slowest function that slow everything down (even in
        # memory usage).
        update_embedding!(Y, grad, vel, gain, lr, i)
    end

    return TriMap{nntype,T}(d, m₁, m₂, r, NN, Y)
end


