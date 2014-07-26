Hessian Eigenmaps
=================

The Hessian Eigenmaps (Hessian LLE, HLLE) method adapts the weights in :doc:`LLE <lle>` to minimize the `Hessian <http://en.wikipedia.org/wiki/Hessian_matrix>`_ operator. Like :doc:`LLE <lle>`, it requires careful setting of the nearest neighbor parameter. The main advantage of Hessian LLE is the only method designed for non-convex data sets [#R1]_.

This package defines a ``HLLE`` type to represent a Hessian LLE results, and provides a set of methods to access its properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``HLLE``, ``n`` be the number of observations, and ``d`` be the output dimension.

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

.. function:: transform(HLLE, X; ...)

    Perform HLLE over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``HLLE``.

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
    # apply HLLE transformation to the dataset
    Y = transform(HLLE, X; k = 12, d = 2)

.. rubric:: References
.. [#R1] Donoho, D. and Grimes, C. "Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data", Proc. Natl. Acad. Sci. USA. 2003 May 13; 100(10): 5591–5596. DOI:`10.1073/pnas.1031596100 <http://dx.doi.org/doi:10.1073/pnas.1031596100>`_
