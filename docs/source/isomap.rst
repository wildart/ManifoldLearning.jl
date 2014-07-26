Isomap
======

`Isomap <http://en.wikipedia.org/wiki/Isomap>`_ is a method for low-dimensional embedding. Isomap is used for computing a quasi-isometric, low-dimensional embedding of a set of high-dimensional data points [#R1]_.

This package defines a ``Isomap`` type to represent a Isomap results, and provides a set of methods to access its properties.

Properties
~~~~~~~~~~~

Let ``M`` be an instance of ``Isomap``, ``n`` be the number of observations, and ``d`` be the output dimension.

.. function:: outdim(M)

    Get the output dimension ``d``, *i.e* the dimension of the subspace.

.. function:: projection(M)

    Get the projection matrix (of size ``(d, n)``). Each column of the projection matrix corresponds to an observation in projected subspace.

.. function:: neighbors(M)

    The number of nearest neighbors used for approximating local coordinate structure.

.. function:: ccomponent(M)

    The observations index array of the largest connected component of the distance matrix.


Data Transformation
~~~~~~~~~~~~~~~~~~~

One can use the ``transform`` method to perform Isomap over a given dataset.

.. function:: transform(Isomap, X; ...)

    Perform Isomap over the data given in a matrix ``X``. Each column of ``X`` is an observation.

    This method returns an instance of ``Isomap``.

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
    # apply Isomap transformation to the dataset
    Y = transform(Isomap, X; k = 12, d = 2)

.. rubric:: References
.. [#R1] Tenenbaum, J. B., de Silva, V. and Langford, J. C. "A Global Geometric Framework for Nonlinear Dimensionality Reduction". Science 290 (5500): 2319-2323, 22 December 2000. `http://isomap.stanford.edu/ <http://isomap.stanford.edu/>`_