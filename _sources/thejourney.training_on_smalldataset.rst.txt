6. Training on Small Dataset
============================

Now the model was trained on ``96x96`` images, so then later we can perform transfer learning to train on ``192x192`` images

LR Range Test
*************

.. figure:: assets/lr-range-test.png


So the max_lr should be about ``0.2-0.4``


Colab was kept alive by using my chrome extension `<https://github.com/satyajitghana/colab-keepalive>`_

Train for 15 Epochs
*******************

**LR Value**
^^^^^^^^^^^^

.. figure:: assets/smalltrain/LR_lr_value.svg

**Train Loss**
^^^^^^^^^^^^^^

.. figure:: assets/smalltrain/BatchLoss_Train_seg_loss.svg
    :width: 70%
    :align: center
    :figclass: align-center

    Train - Segmentation Loss

.. figure:: assets/smalltrain/BatchLoss_Train_depth_loss.svg
    :width: 70%
    :align: center
    :figclass: align-center
    
    Train Depth Loss

**Train Accuracy**
^^^^^^^^^^^^^^^^^^

.. figure:: assets/smalltrain/EpochAccuracy_Train_mIOU.svg
    :width: 70%
    :align: center
    :figclass: align-center

    Train mIOU

.. figure:: assets/smalltrain/EpochAccuracy_Train_mRMSE.svg
    :width: 70%
    :align: center
    :figclass: align-center

    Train mRMSE

**Test Loss**
^^^^^^^^^^^^^

.. figure:: assets/smalltrain/EpochLoss_Test_seg_loss.svg
    :width: 70%
    :align: center
    :figclass: align-center
    
    Test - Segmentation Loss

.. figure:: assets/smalltrain/EpochLoss_Test_depth_loss.svg
    :width: 70%
    :align: center
    :figclass: align-center
    
    Test Depth Loss

**Test Accuracy**
^^^^^^^^^^^^^^^^^

.. figure:: assets/smalltrain/EpochAccuracy_Test_mIOU.svg
    :width: 70%
    :align: center
    :figclass: align-center

    Test mIOU

.. figure:: assets/smalltrain/EpochAccuracy_Test_mRMSE.svg
    :width: 70%
    :align: center
    :figclass: align-center

    Test mRMSE

Results
*******

After first epoch
^^^^^^^^^^^^^^^^^

.. figure:: assets/smalltrain/step1.png


After 4th epoch
^^^^^^^^^^^^^^^

.. figure:: assets/smalltrain/step4.png


After 15th epoch
^^^^^^^^^^^^^^^^

.. figure:: assets/smalltrain/step14.png
