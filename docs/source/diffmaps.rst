Diffusion maps
==============

`Diffusion maps <http://en.wikipedia.org/wiki/Diffusion_map>`_ leverages the relationship between heat diffusion and a random walk; an analogy is drawn between the diffusion operator on a manifold and a Markov transition matrix operating on functions defined on the graph whose nodes were sampled from the manifold [#R1]_.


This package defines a ``DiffMap`` type to represent a Hessian LLE results, and provides a set of methods to access its properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``DiffMap``, ``n`` be the number of observations, and ``d`` be the output dimension.

.. function:: outdim(M)

    Get the output dimension ``d``, *i.e* the dimension of the subspace.

.. function:: projection(M)

    Get the projection matrix (of size ``(d, n)``). Each column of the projection matrix corresponds to an observation in projected subspace.

.. function:: kernel(M)

    The kernel matrix.


Data Transformation
~~~~~~~~~~~~~~~~~~~

One can use the ``transform`` method to perform DiffMap over a given dataset.

.. function:: transform(DiffMap, X; ...)

    Perform DiffMap over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``DiffMap``.

    **Keyword arguments:**

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     d          Output dimension.                                               ``2``
    ----------- --------------------------------------------------------------- ---------------
     t          Number of time steps.                                           ``1``
    ----------- --------------------------------------------------------------- ---------------
     ɛ          The scale parameter.                                            ``1.0``
    =========== =============================================================== ===============


**Example:**

.. code-block:: julia

    using ManifoldLearning

    # suppose X is a data matrix, with each observation in a column
    # apply DiffMap transformation to the dataset
    Y = transform(DiffMap, X; d=2, t=1, ɛ=1.0)

.. rubric:: References
.. [#R1] Coifman, R. & Lafon, S. "Diffusion maps". Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30. DOI:`10.1073/pnas.0500334102 <http://dx.doi.org/doi:10.1073/pnas.0500334102>`_