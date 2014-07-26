Local Tangent Space Alignment
=============================

`Local tangent space alignment <http://en.wikipedia.org/wiki/Local_tangent_space_alignment>`_ (LTSA) is a method for manifold learning, which can efficiently learn a nonlinear embedding into low-dimensional coordinates from high-dimensional data, and can also reconstruct high-dimensional coordinates from embedding coordinates [#R1]_.

This package defines a ``LTSA`` type to represent a LTSA results, and provides a set of methods to access its properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``LTSA``, ``n`` be the number of observations, and ``d`` be the output dimension.

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

One can use the ``transform`` method to perform LTSA over a given dataset.

.. function:: transform(LSTA, X; ...)

    Perform LTSA over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``LTSA``.

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
    # apply LTSA transformation to the dataset
    Y = transform(LTSA, X; k = 12, d = 2)

.. rubric:: References
.. [#R1] Zhang, Zhenyue; Hongyuan Zha. "Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment". SIAM Journal on Scientific Computing 26 (1): 313–338, 2004. DOI:`10.1137/s1064827502419154 <http://dx.doi.org/doi:10.1137/s1064827502419154>`_