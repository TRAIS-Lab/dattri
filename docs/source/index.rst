.. dattri documentation master file, created by
   sphinx-quickstart on Wed Apr  3 12:59:58 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: ../../assets/images/logo.png
   :width: 150
   :align: center

`dattri`: A Library for Efficient Data Attribution
==================================

`dattri` is a PyTorch library for **developing, benchmarking, and deploying efficient data attribution algorithms**. You may use `dattri` to

- Deploy existing data attribution methods to PyTorch models
  - e.g., Influence Function, TracIn, RPS, TRAK, ...
- Develop new data attribution methods with efficient implementation of low-level utility functions
  - e.g., Hessian (HVP/IHVP), Fisher Information Matrix (IFVP), random projection, dropout ensembling, ...
- Benchmark data attribution methods with standard benchmark settings
  - e.g., MNIST-10+LR/MLP, CIFAR-10/2+ResNet-9, MAESTRO + Music Transformer, Shakespeare + nanoGPT, ...

.. toctree::
   :maxdepth: 1
   :caption: Attribution Methods:

   api/task.rst
   api/algorithm.rst

.. toctree::
   :maxdepth: 1
   :caption: Low-level Utility Functions:

   api/hessian.rst
   api/fisher.rst
   api/projection.rst
   api/dropout.rst

.. toctree::
   :maxdepth: 1
   :caption: Benchmark:

   api/benchmark.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
