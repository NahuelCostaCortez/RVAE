# RVAE
A semi-supervised framework to visually assess the progression of time series. In this repository a recurrent version of the VAE is implemented to exploit the generative properties that lead it to learn in an unsupervised way a continuous compressed representation of the data. A classifier is introduced in the VAE training process to control the regulation of the latent space, allowing the network to learn latent variables that set the basis for creating an explainable evaluation of the data. 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NahuelCostaCortez/RVAE/blob/main/FA-RVAE.ipynb)

# Files in this Repository

- FA-RVAE.ipynb: Jupyter notebook with the model implementation and results of a case study.
- data_simulator: data simulation to validate the model performance. Two types of datasets can be generated: sine wave sequences or arrhythmias that reflect different states of AF (Atrial Fibrillation).
- main.py: creation, training and presentation of model results.
- model.py: creation of the model architecture.
- utils.py: helper functions.
- data folder: data to validate the model performance.
- logs folder: training logs record.
- model: saved models applied to the AF dataset to compare the implemented solution with other state-of-the-art classifiers for time series.
