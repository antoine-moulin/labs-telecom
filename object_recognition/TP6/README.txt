In order to work on the TP, you will need to activate an anaconda environment containing the correct packages (Tensorflow and Keras). To do this, type the following commands in a terminal :

export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH
source activate ~anewson/.conda/envs/keras_env


Otherwise, you can create your own environment on your personal computer with the following code :

conda create -n TP
source activate TP
conda config --append channels conda-forge
conda install pip numpy pandas scikit-learn scipy scikit-image matplotlib spyder tensorflow notebook imbalanced-learn nilearn keras

-------------------------------------------------------------

Deadline to finish TP :

- 17/03/2019, 23:59