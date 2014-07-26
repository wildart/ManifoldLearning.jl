Locally Linear Embedding
========================

`Locally Linear Embedding <http://en.wikipedia.org/wiki/Locally_linear_embedding#Locally-linear_embedding>`_ (LLE) technique builds a single global coordinate system of lower dimensionality. By exploiting the local symmetries of linear reconstructions, LLE is able to learn the global structure of nonlinear manifolds [#R1]_.

This package defines a ``LLE`` type to represent a LLE results, and provides a set of methods to access its properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``LLE``, ``n`` be the number of observations, and ``d`` be the output dimension.

.. function:: outdim(M)

    Get the output dimension ``d``, *i.e* the dimension of the subspace.

.. function:: projection(M)

    Get the projection matrix (of size ``(d, n)``). Each column of the projection matrix corresponds to an observation in projected subspace.

.. function:: neighbors(M)

    The number of nearest neighbors used for approximating local coordinate structure.

.. function:: eigvals(M)

    The eigenvalues of alignment matrix.


Data Transformation
~~~~~~~~~~~~~~~~~~~

One can use the ``transform`` method to perform HLLE over a given dataset.

.. function:: transform(LLE, X; ...)

    Perform LLE over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``LLE``.

    **Keyword arguments:**

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     k          The number of nearest neighbors for determining local           ``12``
                coordinate structure.
    ----------- --------------------------------------------------------------- ---------------
     d          Output dimension.                                               ``2``
    =========== =============================================================== ===============


**Example:**

.. code-block:: julia

    using ManifoldLearning

    # suppose X is a data matrix, with each observation in a column
    # apply LLE transformation to the dataset
    Y = transform(LLE, X; k = 12, d = 2)

.. rubric:: References
.. [#R1] Roweis, S. & Saul, L. "Nonlinear dimensionality reduction by locally linear embedding", Science 290:2323 (2000). DOI:`10.1126/science.290.5500.2323 <http://dx.doi.org/doi:10.1126/science.290.5500.2323>`_