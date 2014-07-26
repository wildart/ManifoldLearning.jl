Laplacian Eigenmaps
===================

`Laplacian Eigenmaps <http://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Laplacian_eigenmaps>`_ (LEM) method uses spectral techniques to perform dimensionality reduction. This technique relies on the basic assumption that the data lies in a low-dimensional manifold in a high-dimensional space. The algorithm provides a computationally efficient approach to non-linear dimnsionality reduction that has locally preserving properties [#R1]_.

This package defines a ``LEM`` type to represent a Laplacian Eigenmaps results, and provides a set of methods to access its properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``LEM``, ``n`` be the number of observations, and ``d`` be the output dimension.

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

One can use the ``transform`` method to perform LEM over a given dataset.

.. function:: transform(LEM, X; ...)

    Perform LEM over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``LEM``.

    **Keyword arguments:**

    =========== =============================================================== ===============
      name         description                                                   default
    =========== =============================================================== ===============
     k          The number of nearest neighbors for determining local           ``12``
                coordinate structure.
    ----------- --------------------------------------------------------------- ---------------
     d          Output dimension.                                               ``2``
    ----------- --------------------------------------------------------------- ---------------
     t          The temperature parameters of the heat kernel.                  ``1.0``
    =========== =============================================================== ===============


**Example:**

.. code-block:: julia

    using ManifoldLearning

    # suppose X is a data matrix, with each observation in a column
    # apply Laplacian Eigenmaps transformation to the dataset
    Y = transform(LEM, X; k = 12, d = 2, t = 1.0)

.. rubric:: References
.. [#R1] Belkin, M. and Niyogi, P. "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation". Neural Computation, June 2003; 15 (6):1373-1396. DOI:`10.1162/089976603321780317 <http://dx.doi.org/doi:10.1162/089976603321780317>`_
