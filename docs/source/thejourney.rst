Vathos 🐲 - The Journey
========================

Q  : So what does the model do ?
Ans: It takes in 2D images and outputs segmented image + a depth map, basically recognizes the object for
which it was trained for, segments it out, and also creates a 3D depth map of the image.

Read through The Journey of how the dataset was made, how the model was researched and finally how it was trained,
if you want to see the results of the model, head over to ``Training on Small Dataset``, a few more model outputs can be found
in ``Loss Functions``.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    thejourney.model
    thejourney.dataset.md
    thejourney.loss_functions
    thejourney.testing_hparams_timings
    thejourney.run_on_tpu
    thejourney.training_on_smalldataset
    thejourney.model_exploration_logs.md
