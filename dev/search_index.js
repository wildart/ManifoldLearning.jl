var documenterSearchIndex = {"docs":
[{"location":"diffmap/#Diffusion-maps","page":"Diffusion maps","title":"Diffusion maps","text":"","category":"section"},{"location":"diffmap/","page":"Diffusion maps","title":"Diffusion maps","text":"Diffusion maps leverages the relationship between heat diffusion and a random walk; an analogy is drawn between the diffusion operator on a manifold and a Markov transition matrix operating on functions defined on the graph whose nodes were sampled from the manifold [1].","category":"page"},{"location":"diffmap/","page":"Diffusion maps","title":"Diffusion maps","text":"This package defines a DiffMap type to represent a diffusion map results, and provides a set of methods to access its properties.","category":"page"},{"location":"diffmap/","page":"Diffusion maps","title":"Diffusion maps","text":"DiffMap\nfit(::Type{DiffMap}, X::AbstractArray{T,2}) where {T<:Real}\npredict(R::DiffMap)\nManifoldLearning.kernel(R::DiffMap)","category":"page"},{"location":"diffmap/#ManifoldLearning.DiffMap","page":"Diffusion maps","title":"ManifoldLearning.DiffMap","text":"DiffMap{T <: Real} <: AbstractDimensionalityReduction\n\nThe DiffMap type represents diffusion maps model constructed for T type data.\n\n\n\n\n\n","category":"type"},{"location":"diffmap/#StatsAPI.fit-Union{Tuple{T}, Tuple{Type{DiffMap}, AbstractMatrix{T}}} where T<:Real","page":"Diffusion maps","title":"StatsAPI.fit","text":"fit(DiffMap, data; maxoutdim=2, t=1, α=0.0, ɛ=1.0)\n\nFit a isometric mapping model to data.\n\nArguments\n\ndata::Matrix: a d times nmatrix of observations. Each column of data is\n\nan observation, d is a number of features, n is a number of observations.\n\nKeyword arguments\n\nkernel::Union{Nothing, Function}: the kernel function.\n\nIt maps two input vectors (observations) to a scalar (a metric of their similarity). by default, a Gaussian kernel. If kernel set to nothing, we assume data is instead the n times n  precomputed Gram matrix.\n\nɛ::Real=1.0: the Gaussian kernel variance (the scale parameter). It's ignored if the custom kernel is passed.\nmaxoutdim::Int=2: the dimension of the reduced space.\nt::Int=1: the number of transitions\nα::Real=0.0: a normalization parameter\n\nExamples\n\nX = rand(3, 100)     # toy data matrix, 100 observations\n\n# default kernel\nM = fit(DiffMap, X)  # construct diffusion map model\nR = transform(M)     # perform dimensionality reduction\n\n# custom kernel\nkernel = (x, y) -> x' * y # linear kernel\nM = fit(DiffMap, X, kernel=kernel)\n\n# precomputed Gram matrix\nkernel = (x, y) -> x' * y # linear kernel\nK = ManifoldLearning.pairwise(kernel, eachcol(X), symmetric=true)\nM = fit(DiffMap, K, kernel=nothing)\n\n\n\n\n\n","category":"method"},{"location":"diffmap/#StatsAPI.predict-Tuple{DiffMap}","page":"Diffusion maps","title":"StatsAPI.predict","text":"predict(R::DiffMap)\n\nTransforms the data fitted to the diffusion map model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"diffmap/#ManifoldLearning.kernel-Tuple{DiffMap}","page":"Diffusion maps","title":"ManifoldLearning.kernel","text":"Returns the kernel matrix of the diffusion maps model R\n\n\n\n\n\n","category":"method"},{"location":"diffmap/#References","page":"Diffusion maps","title":"References","text":"","category":"section"},{"location":"diffmap/","page":"Diffusion maps","title":"Diffusion maps","text":"[1]: Coifman, R. & Lafon, S. \"Diffusion maps\". Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30. DOI:10.1073/pnas.0500334102","category":"page"},{"location":"hlle/#Hessian-Eigenmaps","page":"Hessian Eigenmaps","title":"Hessian Eigenmaps","text":"","category":"section"},{"location":"hlle/","page":"Hessian Eigenmaps","title":"Hessian Eigenmaps","text":"The Hessian Eigenmaps (Hessian LLE, HLLE) method adapts the weights in LLE to minimize the Hessian operator. Like LLE, it requires careful setting of the nearest neighbor parameter. The main advantage of Hessian LLE is the only method designed for non-convex data sets [1].","category":"page"},{"location":"hlle/","page":"Hessian Eigenmaps","title":"Hessian Eigenmaps","text":"This package defines a HLLE type to represent a Hessian LLE results, and provides a set of methods to access its properties.","category":"page"},{"location":"hlle/","page":"Hessian Eigenmaps","title":"Hessian Eigenmaps","text":"HLLE\nfit(::Type{HLLE}, X::AbstractArray{T,2}) where {T<:Real}\npredict(R::HLLE)","category":"page"},{"location":"hlle/#ManifoldLearning.HLLE","page":"Hessian Eigenmaps","title":"ManifoldLearning.HLLE","text":"HLLE{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction\n\nThe HLLE type represents a Hessian eigenmaps model constructed for T type data with a help of the NN nearest neighbor algorithm.\n\n\n\n\n\n","category":"type"},{"location":"hlle/#StatsAPI.fit-Union{Tuple{T}, Tuple{Type{HLLE}, AbstractMatrix{T}}} where T<:Real","page":"Hessian Eigenmaps","title":"StatsAPI.fit","text":"fit(HLLE, data; k=12, maxoutdim=2, nntype=BruteForce)\n\nFit a Hessian eigenmaps model to data.\n\nArguments\n\ndata: a matrix of observations. Each column of data is an observation.\n\nKeyword arguments\n\nk: a number of nearest neighbors for construction of local subspace representation\nmaxoutdim: a dimension of the reduced space.\nnntype: a nearest neighbor construction class (derived from AbstractNearestNeighbors)\n\nExamples\n\nM = fit(HLLE, rand(3,100)) # construct Hessian eigenmaps model\nR = predict(M)             # perform dimensionality reduction\n\n\n\n\n\n","category":"method"},{"location":"hlle/#StatsAPI.predict-Tuple{HLLE}","page":"Hessian Eigenmaps","title":"StatsAPI.predict","text":"predict(R::HLLE)\n\nTransforms the data fitted to the Hessian eigenmaps model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"hlle/#References","page":"Hessian Eigenmaps","title":"References","text":"","category":"section"},{"location":"hlle/","page":"Hessian Eigenmaps","title":"Hessian Eigenmaps","text":"[1]: Donoho, D. and Grimes, C. \"Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data\", Proc. Natl. Acad. Sci. USA. 2003 May 13; 100(10): 5591–5596. DOI:10.1073/pnas.1031596100","category":"page"},{"location":"interface/#Programming-interface","page":"Interface","title":"Programming interface","text":"","category":"section"},{"location":"interface/","page":"Interface","title":"Interface","text":"The interface of manifold learning methods in this packages is partially adopted from the packages StatsAPI, MultivariateStats.jl and Graphs.jl. You can implement additional dimensionality reduction algorithms by implementing the following interface.","category":"page"},{"location":"interface/#Dimensionality-Reduction","page":"Interface","title":"Dimensionality Reduction","text":"","category":"section"},{"location":"interface/","page":"Interface","title":"Interface","text":"The following functions are currently available from the interface. NonlinearDimensionalityReduction is an abstract type required for all implemented algorithms models.","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"ManifoldLearning.NonlinearDimensionalityReduction","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"For performing the data dimensionality reduction procedure, a model of the data is constructed by calling fit method, and the transformation of the data given the model is done by predict method.","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"fit(::Type{ManifoldLearning.NonlinearDimensionalityReduction}, X::AbstractMatrix)\npredict(R::ManifoldLearning.NonlinearDimensionalityReduction)","category":"page"},{"location":"interface/#StatsAPI.fit-Tuple{Type{MultivariateStats.NonlinearDimensionalityReduction}, AbstractMatrix}","page":"Interface","title":"StatsAPI.fit","text":"fit(NonlinearDimensionalityReduction, X)\n\nPerform model fitting given the data X\n\n\n\n\n\n","category":"method"},{"location":"interface/#StatsAPI.predict-Tuple{MultivariateStats.NonlinearDimensionalityReduction}","page":"Interface","title":"StatsAPI.predict","text":"predict(R::NonlinearDimensionalityReduction)\n\nReturns a reduced space representation of the data given the model R inform of  the projection matrix (of size (d n)), where d is a dimension of the reduced space and n in the number of the observations. Each column of the projection matrix corresponds to an observation in projected reduced space.\n\n\n\n\n\n","category":"method"},{"location":"interface/","page":"Interface","title":"Interface","text":"There are auxiliary methods that allow to inspect properties of the constructed model.","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"size(R::ManifoldLearning.NonlinearDimensionalityReduction)\neigvals(R::ManifoldLearning.NonlinearDimensionalityReduction)\nvertices(R::ManifoldLearning.NonlinearDimensionalityReduction)\nneighbors(R::ManifoldLearning.NonlinearDimensionalityReduction)","category":"page"},{"location":"interface/#Base.size-Tuple{MultivariateStats.NonlinearDimensionalityReduction}","page":"Interface","title":"Base.size","text":"size(R::NonlinearDimensionalityReduction)\n\nReturns a tuple of the input and reduced space dimensions for the model R\n\n\n\n\n\n","category":"method"},{"location":"interface/#LinearAlgebra.eigvals-Tuple{MultivariateStats.NonlinearDimensionalityReduction}","page":"Interface","title":"LinearAlgebra.eigvals","text":"eigvals(R::NonlinearDimensionalityReduction)\n\nReturns eignevalues of the reduced space reporesentation for the model R\n\n\n\n\n\n","category":"method"},{"location":"interface/#Graphs.vertices-Tuple{MultivariateStats.NonlinearDimensionalityReduction}","page":"Interface","title":"Graphs.vertices","text":"vertices(R::NonlinearDimensionalityReduction)\n\nReturns vertices of largest connected component in the model R.\n\n\n\n\n\n","category":"method"},{"location":"interface/#Graphs.neighbors-Tuple{MultivariateStats.NonlinearDimensionalityReduction}","page":"Interface","title":"Graphs.neighbors","text":"neighbors(R::NonlinearDimensionalityReduction)\n\nReturns the number of nearest neighbors used for aproximate local subspace\n\n\n\n\n\n","category":"method"},{"location":"interface/#Nearest-Neighbors","page":"Interface","title":"Nearest Neighbors","text":"","category":"section"},{"location":"interface/","page":"Interface","title":"Interface","text":"An additional interface is available for creating an implementation of a nearest neighbors algorithm, which is commonly used for dimensionality reduction methods. Use AbstractNearestNeighbors abstract type to derive a type for a new implementation.","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"ManifoldLearning.AbstractNearestNeighbors","category":"page"},{"location":"interface/#ManifoldLearning.AbstractNearestNeighbors","page":"Interface","title":"ManifoldLearning.AbstractNearestNeighbors","text":"AbstractNearestNeighbors\n\nAbstract type for nearest neighbor plug-in implementations.\n\n\n\n\n\n","category":"type"},{"location":"interface/","page":"Interface","title":"Interface","text":"The above interface requires implementation of the following methods:","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"ManifoldLearning.knn(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer) where T<:Real\nManifoldLearning.inradius(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, r::Real) where T<:Real","category":"page"},{"location":"interface/#ManifoldLearning.knn-Union{Tuple{T}, Tuple{ManifoldLearning.AbstractNearestNeighbors, AbstractVecOrMat{T}, Integer}} where T<:Real","page":"Interface","title":"ManifoldLearning.knn","text":"knn(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer; kwargs...) -> (I,D)\n\nReturns (k, n)-matrices of point indexes and distances of k nearest neighbors for points in the (m,n)-matrix X given the NN object.\n\n\n\n\n\n","category":"method"},{"location":"interface/#ManifoldLearning.inradius-Union{Tuple{T}, Tuple{ManifoldLearning.AbstractNearestNeighbors, AbstractVecOrMat{T}, Real}} where T<:Real","page":"Interface","title":"ManifoldLearning.inradius","text":"inradius(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, r::Real; kwargs...) -> (I,D)\n\nReturns collections of point indexes and distances in radius r of points in the (m,n)-matrix X given the NN object.\n\n\n\n\n\n","category":"method"},{"location":"interface/","page":"Interface","title":"Interface","text":"Following auxiliary methods available for any implementation of AbstractNearestNeighbors-derived type:","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"ManifoldLearning.adjacency_list(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer) where T<:Real\nManifoldLearning.adjacency_matrix(NN::ManifoldLearning.AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Integer) where T<:Real","category":"page"},{"location":"interface/#ManifoldLearning.adjacency_list-Union{Tuple{T}, Tuple{ManifoldLearning.AbstractNearestNeighbors, AbstractVecOrMat{T}, Integer}} where T<:Real","page":"Interface","title":"ManifoldLearning.adjacency_list","text":"adjacency_list(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Real; kwargs...) -> (A, W)\n\nPerform construction of an adjacency list A with corresponding weights W from the points in X given the NN object.\n\nIf k is a positive integer, then k nearest neighbors are use for construction.\nIf k is a real number, then radius k neighborhood is used for construction.\n\n\n\n\n\n","category":"method"},{"location":"interface/#ManifoldLearning.adjacency_matrix-Union{Tuple{T}, Tuple{ManifoldLearning.AbstractNearestNeighbors, AbstractVecOrMat{T}, Integer}} where T<:Real","page":"Interface","title":"ManifoldLearning.adjacency_matrix","text":"adjacency_matrix(NN::AbstractNearestNeighbors, X::AbstractVecOrMat{T}, k::Real; kwargs...) -> A\n\nPerform construction of a weighted adjacency distance matrix A from the points in X given the NN object.\n\nIf k is a positive integer, then k nearest neighbors are use for construction.\nIf k is a real number, then radius k neighborhood is used for construction.\n\n\n\n\n\n","category":"method"},{"location":"interface/","page":"Interface","title":"Interface","text":"The default implementation uses inefficient O(n^2) algorithm for nearest neighbors calculations.","category":"page"},{"location":"interface/","page":"Interface","title":"Interface","text":"ManifoldLearning.BruteForce","category":"page"},{"location":"interface/#ManifoldLearning.BruteForce","page":"Interface","title":"ManifoldLearning.BruteForce","text":"BruteForce\n\nCalculate nearest neighborhoods using pairwise distance matrix.\n\n\n\n\n\n","category":"type"},{"location":"isomap/#Isomap","page":"Isomap","title":"Isomap","text":"","category":"section"},{"location":"isomap/","page":"Isomap","title":"Isomap","text":"Isomap is a method for low-dimensional embedding. Isomap is used for computing a quasi-isometric, low-dimensional embedding of a set of high-dimensional data points[1].","category":"page"},{"location":"isomap/","page":"Isomap","title":"Isomap","text":"This package defines a Isomap type to represent a Isomap calculation results, and provides a set of methods to access its properties.","category":"page"},{"location":"isomap/","page":"Isomap","title":"Isomap","text":"Isomap\nfit(::Type{Isomap}, X::AbstractArray{T,2}) where {T<:Real}\npredict(R::Isomap)\npredict(R::Isomap, X::Union{AbstractArray{T,1}, AbstractArray{T,2}}) where T<:Real","category":"page"},{"location":"isomap/#ManifoldLearning.Isomap","page":"Isomap","title":"ManifoldLearning.Isomap","text":"Isomap{NN <: AbstractNearestNeighbors} <: AbstractDimensionalityReduction\n\nThe Isomap type represents an isometric mapping model constructed with a help of the NN nearest neighbor algorithm.\n\n\n\n\n\n","category":"type"},{"location":"isomap/#StatsAPI.fit-Union{Tuple{T}, Tuple{Type{Isomap}, AbstractMatrix{T}}} where T<:Real","page":"Isomap","title":"StatsAPI.fit","text":"fit(Isomap, data; k=12, maxoutdim=2, nntype=BruteForce)\n\nFit an isometric mapping model to data.\n\nArguments\n\ndata: a matrix of observations. Each column of data is an observation.\n\nKeyword arguments\n\nk: a number of nearest neighbors for construction of local subspace representation\nmaxoutdim: a dimension of the reduced space.\nnntype: a nearest neighbor construction class (derived from AbstractNearestNeighbors)\n\nExamples\n\nM = fit(Isomap, rand(3,100)) # construct Isomap model\nR = predict(M)               # perform dimensionality reduction\n\n\n\n\n\n","category":"method"},{"location":"isomap/#StatsAPI.predict-Tuple{Isomap}","page":"Isomap","title":"StatsAPI.predict","text":"predict(R::Isomap)\n\nTransforms the data fitted to the Isomap model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"isomap/#StatsAPI.predict-Union{Tuple{T}, Tuple{Isomap, AbstractVecOrMat{T}}} where T<:Real","page":"Isomap","title":"StatsAPI.predict","text":"predict(R::Isomap, X::AbstractVecOrMat)\n\nReturns a transformed out-of-sample data X given the Isomap model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"isomap/#References","page":"Isomap","title":"References","text":"","category":"section"},{"location":"isomap/","page":"Isomap","title":"Isomap","text":"[1]: Tenenbaum, J. B., de Silva, V. and Langford, J. C. \"A Global Geometric Framework for Nonlinear Dimensionality Reduction\". Science 290 (5500): 2319-2323, 22 December 2000.","category":"page"},{"location":"lem/#Laplacian-Eigenmaps","page":"Laplacian Eigenmaps","title":"Laplacian Eigenmaps","text":"","category":"section"},{"location":"lem/","page":"Laplacian Eigenmaps","title":"Laplacian Eigenmaps","text":"Laplacian Eigenmaps (LEM) method uses spectral techniques to perform dimensionality reduction. This technique relies on the basic assumption that the data lies in a low-dimensional manifold in a high-dimensional space. The algorithm provides a computationally efficient approach to non-linear dimensionality reduction that has locally preserving properties [1].","category":"page"},{"location":"lem/","page":"Laplacian Eigenmaps","title":"Laplacian Eigenmaps","text":"This package defines a LEM type to represent a Laplacian eigenmaps results, and provides a set of methods to access its properties.","category":"page"},{"location":"lem/","page":"Laplacian Eigenmaps","title":"Laplacian Eigenmaps","text":"LEM\nfit(::Type{LEM}, X::AbstractArray{T,2}) where {T<:Real}\npredict(R::LEM)","category":"page"},{"location":"lem/#ManifoldLearning.LEM","page":"Laplacian Eigenmaps","title":"ManifoldLearning.LEM","text":"LEM{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction\n\nThe LEM type represents a Laplacian eigenmaps model constructed for T type data with a help of the NN nearest neighbor algorithm.\n\n\n\n\n\n","category":"type"},{"location":"lem/#StatsAPI.fit-Union{Tuple{T}, Tuple{Type{LEM}, AbstractMatrix{T}}} where T<:Real","page":"Laplacian Eigenmaps","title":"StatsAPI.fit","text":"fit(LEM, data; k=12, maxoutdim=2, ɛ=1.0, nntype=BruteForce)\n\nFit a Laplacian eigenmaps model to data.\n\nArguments\n\ndata: a matrix of observations. Each column of data is an observation.\n\nKeyword arguments\n\nk: a number of nearest neighbors for construction of local subspace representation\nmaxoutdim: a dimension of the reduced space.\nnntype: a nearest neighbor construction class (derived from AbstractNearestNeighbors)\nɛ: a Gaussian kernel variance (the scale parameter)\nlaplacian: a form of the Laplacian matrix used for spectral decomposition\n:unnorm: an unnormalized Laplacian\n:sym: a symmetrically normalized Laplacian\n:rw: a random walk normalized Laplacian\n\nExamples\n\nM = fit(LEM, rand(3,100)) # construct Laplacian eigenmaps model\nR = predict(M)          # perform dimensionality reduction\n\n\n\n\n\n","category":"method"},{"location":"lem/#StatsAPI.predict-Tuple{LEM}","page":"Laplacian Eigenmaps","title":"StatsAPI.predict","text":"predict(R::LEM)\n\nTransforms the data fitted to the Laplacian eigenmaps model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"lem/#References","page":"Laplacian Eigenmaps","title":"References","text":"","category":"section"},{"location":"lem/","page":"Laplacian Eigenmaps","title":"Laplacian Eigenmaps","text":"[1]: Belkin, M. and Niyogi, P. \"Laplacian Eigenmaps for Dimensionality Reduction and Data Representation\". Neural Computation, June 2003; 15 (6):1373-1396. DOI:10.1162/089976603321780317","category":"page"},{"location":"ltsa/#Local-Tangent-Space-Alignment","page":"Local Tangent Space Alignment","title":"Local Tangent Space Alignment","text":"","category":"section"},{"location":"ltsa/","page":"Local Tangent Space Alignment","title":"Local Tangent Space Alignment","text":"Local tangent space alignment (LTSA) is a method for manifold learning, which can efficiently learn a nonlinear embedding into low-dimensional coordinates from high-dimensional data, and can also reconstruct high-dimensional coordinates from embedding coordinates [1].","category":"page"},{"location":"ltsa/","page":"Local Tangent Space Alignment","title":"Local Tangent Space Alignment","text":"This package defines a LTSA type to represent a local tangent space alignment results, and provides a set of methods to access its properties.","category":"page"},{"location":"ltsa/","page":"Local Tangent Space Alignment","title":"Local Tangent Space Alignment","text":"LTSA\nfit(::Type{LTSA}, X::AbstractArray{T,2}) where {T<:Real}\npredict(R::LTSA)","category":"page"},{"location":"ltsa/#ManifoldLearning.LTSA","page":"Local Tangent Space Alignment","title":"ManifoldLearning.LTSA","text":"LTSA{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction\n\nThe LTSA type represents a local tangent space alignment model constructed for T type data with a help of the NN nearest neighbor algorithm.\n\n\n\n\n\n","category":"type"},{"location":"ltsa/#StatsAPI.fit-Union{Tuple{T}, Tuple{Type{LTSA}, AbstractMatrix{T}}} where T<:Real","page":"Local Tangent Space Alignment","title":"StatsAPI.fit","text":"fit(LTSA, data; k=12, maxoutdim=2, nntype=BruteForce)\n\nFit a local tangent space alignment model to data.\n\nArguments\n\ndata: a matrix of observations. Each column of data is an observation.\n\nKeyword arguments\n\nk: a number of nearest neighbors for construction of local subspace representation\nmaxoutdim: a dimension of the reduced space.\nnntype: a nearest neighbor construction class (derived from AbstractNearestNeighbors)\n\nExamples\n\nM = fit(LTSA, rand(3,100)) # construct LTSA model\nR = transform(M)           # perform dimensionality reduction\n\n\n\n\n\n","category":"method"},{"location":"ltsa/#StatsAPI.predict-Tuple{LTSA}","page":"Local Tangent Space Alignment","title":"StatsAPI.predict","text":"predict(R::LTSA)\n\nTransforms the data fitted to the local tangent space alignment model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"ltsa/#References","page":"Local Tangent Space Alignment","title":"References","text":"","category":"section"},{"location":"ltsa/","page":"Local Tangent Space Alignment","title":"Local Tangent Space Alignment","text":"[1]: Zhang, Zhenyue; Hongyuan Zha. \"Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment\". SIAM Journal on Scientific Computing 26 (1): 313–338, 2004. DOI:10.1137/s1064827502419154","category":"page"},{"location":"#ManifoldLearning.jl","page":"Home","title":"ManifoldLearning.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package ManifoldLearning aims to provide a library for manifold learning and non-linear dimensionality reduction. It provides set of nonlinear dimensionality reduction methods, such as Isomap, LLE, LTSA, and etc.","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install the package just type","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add ManifoldLearning","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Plots\ngr(fmt=:svg)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The following example shows how to apply Isomap dimensionality reduction method to the build-in S curve dataset.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ManifoldLearning\nX, L = ManifoldLearning.scurve(segments=5);\nscatter3d(X[1,:], X[2,:], X[3,:], c=L,palette=cgrad(:default),ms=2.5,leg=:none,camera=(10,10))","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now, we perform dimensionality reduction procedure and plot the resulting dataset:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Y = predict(fit(Isomap, X))\nscatter(Y[1,:], Y[2,:], c=L, palette=cgrad(:default), ms=2.5, leg=:none)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Following dimensionality reduction methods are implemented in this package:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Methods Description\nIsomap Isometric mapping\nLLE Locally Linear Embedding\nHLLE Hessian Eigenmaps\nLEM Laplacian Eigenmaps\nLTSA Local Tangent Space Alignment\nDiffMap Diffusion maps","category":"page"},{"location":"","page":"Home","title":"Home","text":"Notes: All methods implemented in this package adopt the column-major convention of JuliaStats: in a data matrix, each column corresponds to a sample/observation, while each row corresponds to a feature (variable or attribute).","category":"page"},{"location":"lle/#Locally-Linear-Embedding","page":"Locally Linear Embedding","title":"Locally Linear Embedding","text":"","category":"section"},{"location":"lle/","page":"Locally Linear Embedding","title":"Locally Linear Embedding","text":"Locally Linear Embedding (LLE) technique builds a single global coordinate system of lower dimensionality. By exploiting the local symmetries of linear reconstructions, LLE is able to learn the global structure of nonlinear manifolds [1].","category":"page"},{"location":"lle/","page":"Locally Linear Embedding","title":"Locally Linear Embedding","text":"This package defines a LLE type to represent a LLE results, and provides a set of methods to access its properties.","category":"page"},{"location":"lle/","page":"Locally Linear Embedding","title":"Locally Linear Embedding","text":"LLE\nfit(::Type{LLE}, X::AbstractArray{T,2}) where {T<:Real}\npredict(R::LLE)","category":"page"},{"location":"lle/#ManifoldLearning.LLE","page":"Locally Linear Embedding","title":"ManifoldLearning.LLE","text":"LLE{NN <: AbstractNearestNeighbors, T <: Real} <: NonlinearDimensionalityReduction\n\nThe LLE type represents a locally linear embedding model constructed for T type data constructed with a help of the NN nearest neighbor algorithm.\n\n\n\n\n\n","category":"type"},{"location":"lle/#StatsAPI.fit-Union{Tuple{T}, Tuple{Type{LLE}, AbstractMatrix{T}}} where T<:Real","page":"Locally Linear Embedding","title":"StatsAPI.fit","text":"fit(LLE, data; k=12, maxoutdim=2, nntype=BruteForce, tol=1e-5)\n\nFit a locally linear embedding model to data.\n\nArguments\n\ndata: a matrix of observations. Each column of data is an observation.\n\nKeyword arguments\n\nk: a number of nearest neighbors for construction of local subspace representation\nmaxoutdim: a dimension of the reduced space.\nnntype: a nearest neighbor construction class (derived from AbstractNearestNeighbors)\ntol: an algorithm regularization tolerance\n\nExamples\n\nM = fit(LLE, rand(3,100)) # construct LLE model\nR = transform(M)          # perform dimensionality reduction\n\n\n\n\n\n","category":"method"},{"location":"lle/#StatsAPI.predict-Tuple{LLE}","page":"Locally Linear Embedding","title":"StatsAPI.predict","text":"predict(R::LLE)\n\nTransforms the data fitted to the LLE model R into a reduced space representation.\n\n\n\n\n\n","category":"method"},{"location":"lle/#References","page":"Locally Linear Embedding","title":"References","text":"","category":"section"},{"location":"lle/","page":"Locally Linear Embedding","title":"Locally Linear Embedding","text":"[1]: Roweis, S. & Saul, L. \"Nonlinear dimensionality reduction by locally linear embedding\", Science 290:2323 (2000). DOI:10.1126/science.290.5500.2323","category":"page"},{"location":"datasets/#Datasets","page":"Datasets","title":"Datasets","text":"","category":"section"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"using Plots, ManifoldLearning\ngr(fmt=:svg)","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"The ManifoldLearning package provides an implementation of synthetic test datasets:","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"ManifoldLearning.swiss_roll","category":"page"},{"location":"datasets/#ManifoldLearning.swiss_roll","page":"Datasets","title":"ManifoldLearning.swiss_roll","text":"swiss_roll(n::Int, noise::Real=0.03, segments=1)\n\nGenerate a swiss roll dataset of n points with point coordinate noise variance, and partitioned on number of segments.\n\n\n\n\n\n","category":"function"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"X, L = ManifoldLearning.swiss_roll(segments=5); #hide\nscatter3d(X[1,:], X[2,:], X[3,:], c=L.+2, palette=cgrad(:default), ms=2.5, leg=:none, camera=(10,10)) #hide","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"ManifoldLearning.spirals","category":"page"},{"location":"datasets/#ManifoldLearning.spirals","page":"Datasets","title":"ManifoldLearning.spirals","text":"spirals(n::Int, noise::Real=0.03, segments=1)\n\nGenerate a spirals dataset of n points with point coordinate noise variance.\n\n\n\n\n\n","category":"function"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"X, L = ManifoldLearning.spirals(segments=5); #hide\nscatter3d(X[1,:], X[2,:], X[3,:], c=L.+2, palette=cgrad(:default), ms=2.5, leg=:none, camera=(10,10)) #hide","category":"page"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"ManifoldLearning.scurve","category":"page"},{"location":"datasets/#ManifoldLearning.scurve","page":"Datasets","title":"ManifoldLearning.scurve","text":"scurve(n::Int, noise::Real=0.03, segments=1)\n\nGenerate an S curve dataset of n points with point coordinate noise variance.\n\n\n\n\n\n","category":"function"},{"location":"datasets/","page":"Datasets","title":"Datasets","text":"X, L = ManifoldLearning.scurve(segments=5); #hide\nscatter3d(X[1,:], X[2,:], X[3,:], c=L.+2, palette=cgrad(:default), ms=2.5, leg=:none, camera=(10,10)) #hide","category":"page"}]
}
