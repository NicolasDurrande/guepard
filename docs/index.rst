Guepard: a library for Gaussian Process Ensembles
================================================================

Guepard aims at making Gaussian Process (GP) models faster and amenable to large datasets up to a few million points. It relies on ensembles of GP models where the sub-models are trained on subsets of the data. It is built on top of Tensorflow_ / GPflow_ and implements various rules for aggregating the sub-model predictions into a single distribution:

* Equivalent Observation as described in the AISTATS submission
* Nested GPs [Rullière 2018]
* Barycenter GP [Cohen 2020]
* Several classic baselines: (generalised) Product of Expert, (robust) Bayesian committee machine, etc.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notebooks/explaining_equivalent_obs
   notebooks/equivalent_obs_GPR_ensembles
   notebooks/equivalent_obs_SVGP_ensembles

.. toctree::
   :maxdepth: 1
   :caption: Practical information

   installation
   developer


.. toctree::
   :maxdepth: 1
   :caption: API
   :titlesonly:

   reference <autoapi/guepard/index>


.. _GPflow: https://github.com/GPflow/GPflow
.. _Tensorflow: https://github.com/tensorflow/tensorflow