vathos.model.loss
=================

.. currentmodule:: vathos.model.loss

The Various Loss Functions that can be used with the model

.. note::
    All the loss functions take the Logits as input, internally they will sigmoid the input
    perform their operations and then return the mean error

Segmentation Loss
-----------------

.. autoclass:: DiceLoss

.. autoclass:: BCEDiceLoss

.. autoclass:: TverskyLoss

.. autoclass:: BCETverskyLoss

Depth Loss
----------

.. autoclass:: RMSELoss

.. autoclass:: BerHuLoss

.. autoclass:: GradLoss

.. autoclass:: SSIMLoss

.. autoclass:: RMSEwSSIMLoss

Accuracy Functions
------------------

.. autofunction:: iou

.. autofunction:: rmse
