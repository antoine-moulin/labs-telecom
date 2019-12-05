Deadline to finish TP :

- 03/04/2019, 23:59

---------------------------------------------

In order to work on the TP, you will need to activate an anaconda environment containing the correct packages (Tensorflow, in particular). To do this, type the following commands in a terminal :


export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH
source activate ~anewson/.conda/envs/tf_env


Or:

export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH
conda create -n TP
source activate TP
conda install pip numpy pandas scikit-learn scipy scikit-image matplotlib tensorflow notebook 
conda install -c conda-forge imbalanced-learn nilearn (optional)

If you need more space, you can look for the biggest files using:

find /cal/homes/USERNAME -name '*' -size +10M


