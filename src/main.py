# Imported modules
from config import params
from init_models import init_model
from data_mgmt import prepare_data
from model_training import train_model
from model_testing import test_model, test_ensemble
from create_explanation import create_all_cams, create_average_cam, concat_all_cams
from create_graphs import create_graphs


def run(
        seed=params["seed"],
        model_name=None,
        model_explanation=None,
        augmented_data=False,
        load_models=False,
        train_models=False,
        create_cams=False,
        average_cams=False,
        concat_cams=False,
        test_models=False,
        test_model_ensemble=False,
        num_runs=10,
        explanation_type=None
    ):
    """
    Main script to execute various actions based on specified parameters, including training, testing,
    creating Grad-CAM activation maps, and more.

    Args:
        seed (int): Random seed for reproducibility.
        model_name (str, optional): The name of the model to initialize or load. Default is None.
        model_explanation (str, optional): The name of the model providing explanations, e.g., 'densenet161'. Default is None.
        augmented_data (bool): Indicates whether to use augmented data. Default is False.
        load_models (bool): Flag indicating whether to load existing model parameters from a saved state dictionary. Default is False.
        train_models (bool): Flag indicating whether to train the models. Default is False.
        create_cams (bool): Flag indicating whether to create Grad-CAMs for the images. Default is False.
        average_cams (bool): Flag indicating whether to average multiple Grad-CAMs to create an average Grad-CAM. Default is False.
        concat_cams (bool): Flag indicating whether to concatenate all Grad-CAMs into one tensor. Default is False.
        test_models (bool): Flag indicating whether to test a single model. Default is False.
        test_model_ensemble (bool): Flag indicating whether to test an ensemble of models. Default is False.
        num_runs (int): Number of runs from which to measure performance metrics and create confidence intervals if model testing is specified. Default is 10.
        explanation_type (str): The type of explanation to be used if model testing is specified. Default is None.
    """


    # Perform training and validation (multiple epochs) if specified
    if train_models:

        # Initializing model
        model = init_model(
            model_name=model_name,
            augmented_data=augmented_data,
            load_models=load_models,
            num_extra_channels=5 if model_explanation == "concatenated" else 1
        )
        model = model.to(params["device"])
        
        # Initializing dataset
        _, _, _, train_dataloader, val_dataloader, _ = prepare_data(
            seed=seed,
            augmented_data=augmented_data,
            model_explanation=model_explanation,
            split=True
        )

        # Perform model training
        train_metrics, val_metrics = train_model(
            seed=seed,
            model=model,
            model_name=model_name,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            augmented_data=augmented_data,
            save_model=True
        )
        
        # Create graphs of training and validation metrics
        create_graphs(
            model_name=model_name,
            augmented_data=augmented_data,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )

    # If specified, create CAMs from specified model
    if create_cams:

        create_all_cams(
            load_models=load_models
        )

    # Average over all created CAMs to create average CAM if specified
    if average_cams:
        create_average_cam()

    # Concatenate all created CAMs into one tensor if specified
    if concat_cams:
        concat_all_cams()

    # Perform model testing if specified
    if test_models:

        test_model(
            model_name=model_name,
            augmented_data=augmented_data,
            num_runs=num_runs,
            explanation_type=explanation_type,
            peer_model_name=model_explanation
        )

    # Perform ensemble testing if specified
    if test_model_ensemble:
        
        test_ensemble(
            augmented_data=augmented_data,
            num_runs=num_runs,
            explanation_type=explanation_type,
            peer_model_name=model_explanation
        )

if __name__ == "__main__":
    run(model_name="densenet161", train_models=True, create_cams=True)