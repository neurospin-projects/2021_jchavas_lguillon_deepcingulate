SimCLR training
#########

Configuration
========

In the configuration file configs/dataset/cingulate.yaml, update the path of the pickle file and of the csv file:

.. code-block:: shell

    pickle_normal: /path/to/pickle_file
    train_val_csv_file: /path/to/csv/file

Training
=====

We launch the training from SimCLR folder:

.. code-block:: shell

    python3 main.py

It will save the file in the folder ../../../Output/YYY-MM-DD/HH-MM-SS

Evaluate results
================

We evaluate the results by scanning the output deep learning folders and reading the test csv file:

.. code-block:: shell

    python3 synthesize_results -s /path/to/output/file -c /path/to/csv/file

