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

It will save the training files in the folder ../../../Output/YYYY-MM-DD/HH-MM-SS (YYYY, MM and DD being respectively the year, the maoth and the day; HH, MM and SS are the hour, the minutes and the seconds of xhen the training was launched).

To follow in real time the training, we can go to the output folder YYYY-MM-DD and use tensordboard as follows:

.. code-block:: shell

    cd path/to/YYYY-MM-DD
    tensorboard --logdir .


Evaluate results
================

We evaluate the results by scanning the output deep learning folders and reading the test csv file:

.. code-block:: shell

    python3 synthesize_results -s /path/to/output/folder -c /path/to/csv/file

The /path/to/output/folder is typically the path to the folder YYYY-MM-DD created above and containing one or more training output folders ad subfolders.
