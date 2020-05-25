5. Run on TPU !
===============

Github Link : `<https://github.com/satyajitghana/ProjektDepth/blob/master/notebooks/09_DepthModel_Experiments_TPU.ipynb>`_
Colab Link  : `<https://colab.research.google.com/github/satyajitghana/ProjektDepth/blob/master/notebooks/09_DepthModel_Experiments_TPU.ipynb>`_

Refer to this article: `<https://medium.com/@satyajitghana7/speed-up-your-model-training-w-tpu-on-google-colab-c55ac0f634d9>`_

Running on TPU almost halfed the training time, but at the time of writing this documentation, the code doesn't work, or is buggy since
the PyTorch-XLA package is being updated frequently and there has been a lot of changes that's being introduced, which is the reason i decided
to stay away from it when actually training my model. But its good for small trains.
