
Unsupervised Representation Learning of Cingulate Cortical Folding Patterns
------------

Official Pytorch implementation for Unsupervised Learning and Cortical Folding `paper <(https://openreview.net/forum?id=ueRZzvQ_K6u>`_.
The project aims to study cortical folding patterns thanks to unsupervised deep learning methods.

.. image:: images/pipeline.png
.. image:: images/clustering.png
.. image:: images/ma.png

Dependencies
-----------
- python >= 3.6
- pytorch >= 1.4.0
- numpy >= 1.16.6
- pandas >= 0.23.3


Installation
------------

.. code-block:: shell

    git clone https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate
    cd 2021_jchavas_lguillon_deepcingulate
    pip3 install -e .
    
Training the models
-------------------
We follow the README for each model:
    * :ref: `betaVAE/README.md`
    * :ref: `SimCLR/README.rst`
