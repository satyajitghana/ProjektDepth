vathos.data_loader
==================

.. currentmodule:: vathos.data_loader

The Dataloader classes that can be used with vathos

Some DataVisualization

BG Images

.. image:: assets/dataset-bg.png

FG_BG Images

.. image:: assets/dataset-fg_bg.png

FG_BG_MASK Images

.. image:: assets/dataset-fg_bg_mask.png

DEPTH_FG_BG Images

.. image:: assets/dataset-depth_fg_bg.png


Mean and Standard Deviation

.. code-block:: none

    bg_stat = (['0.573435604572296', '0.520844697952271', '0.457784473896027'], ['0.207058250904083', '0.208138316869736', '0.215291306376457'])
    fg_bg_stat = (['0.568499565124512', '0.512103974819183', '0.452332496643066'], ['0.211068645119667', '0.211040720343590', '0.216081097722054'])
    fg_bg_mask_stat = (['0.062296919524670', '0.062296919524670', '0.062296919524670'], ['0.227044790983200', '0.227044790983200', '0.227044790983200'])
    depth_fg_bg_stat = (['0.302973538637161', '0.302973538637161', '0.302973538637161'], ['0.101284727454185', '0.101284727454185', '0.101284727454185'])

.. autoclass:: DenseDepth
    :members:
