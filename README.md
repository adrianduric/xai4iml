# XAug - Explanation Augmentation

This repository contains code and other material related to the Explanation Augmentation: Knowledge Sharing Through Explainable Artificial Intelligence paper.

## Contents

The repository contains the source code used to generate the results presented in the paper, as well as files containing all the collected metric scores from testing the proposed methods on different models. The following is a description of the content in the source files:

- `main.py`: Main script used to run code from the other source files.
- `config.py`: Contains configuration parameters used in the other source files, including model hyperparameters.
- `data_mgmt.py`: Contains numerical labels of each class in the HyperKvasir dataset, and all the data transforms used for training and evaluation of both the standard teacher models and the XAug student models. Also contains the HyperKvasirDataset class (extends the PyTorch Dataset class) which is used to contain HyperKvasir images, as well as a function to prepare DataLoaders for training, evaluation and explanation generation (see `create_explanation.py`).
- `init_models.py`: Contains functions used for initializing the models used in the projects, both standard teacher models and XAug student models. Also contains helper functions used to modify the standard models into accepting XAug data, and for adapting the last layer for classification on the HyperKvasir dataset.
- `model_training.py`: Contains functions for one-epoch training and evaluation of a model, and a function for complete training of a model.
- `model_testing.py`: Contains functions for complete training and evaluation of a given model and of an ensemble model containing all the models used in the project. Also contains a helper function for calculating confidence intervals, used for reporting final results.
- `create_explanation.py`: Contains functions for generating Grad-CAM explanations of all images in the HyperKvasir dataset using all models in the project, upsampling and appending them as a new channel to the original image, and saving them as a separate dataset. Also contains functions for per-image concatenation and averaging of these explanations, as well as a helper function for selecting the appropriate layer of the models used in the project to generate Grad-CAM explanations from.
- `compute_metrics.py`: Contains the TorchMetrics objects used to calculate metric scores during model training and evaluation, and functions for performing the calculation.
- `create_graphs.py`: Contains a function for generating plots from the obtained metric scores. Can be used in both training and evaluation.
