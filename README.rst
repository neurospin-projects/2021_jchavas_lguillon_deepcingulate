
Introduction
------------

The project aims to study cortical folding patterns thanks to deep learning tools.
Project and API documentation can be found at: `https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate <https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate>`_.

Development
-----------

.. code-block:: shell

    git clone https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate

    # Install for development
    bv bash
    cd 2021_jchavas_lguillon_deepcingulate
    virtualenv --python=python3 --system-site-packages venv
    . venv/bin/activate
    pip3 install -e .

    # Tests
    python3 -m pytest  # run tests

Notebooks are in the repertory notebooks. We access notebooks using:

.. code-block:: shell

    bv bash # to enter brainvisa environnment
    jupyter notebook # then click on file to open a notebook

