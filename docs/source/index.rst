
Welcome to ManifoldLearning's documentation
=============================================

*ManifoldLearning.jl* is a Julia package for manifold learning and non-linear dimensionality reduction. It proides set of nonlinear dimensionality reduction methods, such as :doc:`Isomap <isomap>`, :doc:`LLE <lle>`, :doc:`LTSA <ltsa>`, etc.

**Methods:**

.. toctree::
   :maxdepth: 1

   isomap.rst
   diffmaps.rst
   lem.rst
   lle.rst
   hlle.rst
   ltsa.rst

**Notes:**

All methods implemented in this package adopt the column-major convention of JuliaStats: in a data matrix, each column corresponds to a sample/observation, while each row corresponds to a feature (variable or attribute).