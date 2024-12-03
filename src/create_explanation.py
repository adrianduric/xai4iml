# Imported modules
from config import params
from init_models import init_model
from data_mgmt import prepare_data

import os
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from tqdm import tqdm


def get_last_non_classification_layer(model):
    """
    Returns the last non-classification layer of the given model.
    
    Args:
        model (torch.nn.Module): The pre-trained model whose last non-classification layer is to be returned.
    
    Returns:
        torch.nn.Module: The last non-classification layer of the model.
    
    Raises:
        ValueError: If the provided model architecture is unsupported.
    """
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    elif isinstance(model, models.VGG) or isinstance(model, models.AlexNet):
        return model.features[-1]
    elif isinstance(model, models.DenseNet):
        return model.features[-1]
    elif isinstance(model, models.MobileNetV2) or isinstance(model, models.MobileNetV3):
        return model.features[-1]
    elif isinstance(model, models.VisionTransformer):
        return model.encoder.layers[-1].ln_1
    elif isinstance(model, models.SwinTransformer):
        return model.features[-1][-1].norm1
    else:
        raise ValueError("Unsupported model architecture!")


def create_all_cams(load_models):
    """
    Creates Grad-CAM activation maps for all images in the dataloader using all specified models.
    The CAMs are appended to the original images as an additional channel and saved as tensors.
    
    Args:
        load_models (bool): Flag indicating whether to load saved models from memory or not.
    """

    # Names of all models to be included
    model_names = params["model_names"]

    # Making CAMS from each model
    for model_name in model_names:

        # Initializing model
        model = init_model(
            model_name=model_name,
            augmented_data=False,
            load_models=load_models,
            num_extra_channels=None
        )
        model = model.to(params["device"])

        # Initializing dataset
        _, complete_dataloader = prepare_data(
            seed=None,
            augmented_data=False,
            model_explanation=None,
            split=False,
            batch_size=1 if isinstance(model, models.VisionTransformer) else 8 # To handle issue where create_all_cams does not work with minibatches for ViT, use batch_size=1 if model is ViT
        )

        model.eval()

        # Getting last non-classification layer to create CAM with
        target_layers = [get_last_non_classification_layer(model=model)]

        # Defining reshape_transform for ViT
        def reshape_transform_vit(tensor, height=14, width=14):
            result = tensor[:, 1 :  , :].reshape(tensor.size(0),
                height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        
        # Defining reshape_transform for Swin and Swin_V2
        def reshape_transform_swin(tensor, height=7, width=7):
            # Bring the channels to the first dimension,
            # like in CNNs.
            result = tensor.transpose(2, 3).transpose(1, 2)
            return result

        # Choosing reshape_transform
        if isinstance(model, models.VisionTransformer):
            transform = reshape_transform_vit

        elif isinstance(model, models.SwinTransformer):
            transform = reshape_transform_swin
        else:
            transform = None

        # Initialize Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=transform)

        print("Creating CAM explanations for all images...")

        for X, y, paths, idxs in tqdm(complete_dataloader):

            # Loading batch to device
            X = X.to(params["device"])
            y = y.to(params["device"])

            if params["num_classes"] > 2:
                targets = [ClassifierOutputTarget(label) for label in y]
            else:
                targets = [BinaryClassifierOutputTarget(label) for label in y]

            grayscale_cam = cam(input_tensor=X, targets=targets)

            # Append CAM to original image as additional channel
            cam_tensor = torch.tensor(grayscale_cam).to(params["device"]).unsqueeze(dim=1)
            X_with_cam = torch.cat((X, cam_tensor), dim=1)

            # Saving each image tensor to a separate file
            for i in range(X.shape[0]):
                
                image_sample = X_with_cam[i].clone()

                # Changing original file path to separate folder for augmented images
                new_path = paths[i].replace("/hyper-kvasir", f"/augmented_images/hyper-kvasir/{model_name}")

                # Changing file format to indicate PyTorch tensor, not RGB image
                new_path = new_path.replace(".jpg", ".pt")

                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                torch.save(image_sample, new_path)

    print("CAMs complete.")

def create_average_cam():
    """
    Averages over all created Grad-CAM activation maps to generate an average Grad-CAM for each image and saves the results.
    """

    # Names of all models to be included
    model_names = params["model_names"]

    # Iterate over all images
    csv_file_path = os.path.join(os.getcwd(), "res/augmented_images/hyper-kvasir")

    with open(os.path.join(csv_file_path, "image-labels.csv"), "r") as file:
        
        file.readline()

        for line in file:

            img_filename, organ, finding, classification = line.strip().split(",")

            img_filename += ".pt"
            organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

            relative_file_path = os.path.join(
                organ, classification, finding, img_filename
            )

            # List to collect all model explanations for current image
            all_cams = []

            # Iterate over all model explanations
            for model_name in model_names:
                
                img_dir_path = os.path.join(os.getcwd(), f"res/augmented_images/hyper-kvasir/{model_name}")

                path_to_img_file = os.path.join(
                    img_dir_path, relative_file_path
                )

                # Load CAM produced from current model, and store in list
                current_cam = torch.load(path_to_img_file)
                all_cams.append(current_cam)

            # Copy the RGB channels from the original image
            rgb_channels = all_cams[0][:3] # Shape [3, 224, 224]
                
            # Stack the 4th channels (CAMs) of all tensors along the channel dimension
            all_cams_stacked = torch.stack([cam[3] for cam in all_cams], dim=0)  # Shape [#cams, 224, 224]

            # Average the stacked CAMs along the first dimension
            average_cam = torch.mean(all_cams_stacked, dim=0)  # Shape [224, 224]

            # Combine the first 3 channels with the averaged 4th channel
            average_cam_image = torch.cat((rgb_channels, average_cam.unsqueeze(0)), dim=0)  # Shape [4, 224, 224]

            # Asserting that the first 3 channels are the same as in the original RGB image
            for cam_image in all_cams:
                assert torch.equal(cam_image[:3], average_cam_image[:3]), "The first 3 channels are not the same"
            
            # Saving average CAM
            save_path = os.path.join(os.getcwd(), "res/augmented_images/hyper-kvasir/average/", relative_file_path)
                                         
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(average_cam_image, save_path)

def concat_all_cams():
    """
    Concatenates all created Grad-CAMs with the original RGB channels for each image and saves the results.
    """

    # Names of all models to be included
    model_names = params["model_names"]
    
    # Iterate over all images
    csv_file_path = os.path.join(os.getcwd(), "res/augmented_images/hyper-kvasir")

    with open(os.path.join(csv_file_path, "image-labels.csv"), "r") as file:
        
        file.readline()

        for line in file:

            img_filename, organ, finding, classification = line.strip().split(",")

            img_filename += ".pt"
            organ = "lower-gi-tract" if organ == "Lower GI" else "upper-gi-tract"

            relative_file_path = os.path.join(
                organ, classification, finding, img_filename
            )

            # List to collect all model explanations for current image
            all_cams = []

            # Iterate over all model explanations
            for model_name in model_names:
                
                img_dir_path = os.path.join(os.getcwd(), f"res/augmented_images/hyper-kvasir/{model_name}")

                path_to_img_file = os.path.join(
                    img_dir_path, relative_file_path
                )

                # Load CAM produced from current model, and store in list
                current_cam = torch.load(path_to_img_file)
                all_cams.append(current_cam)

            # Copy the RGB channels from the original image
            rgb_channels = all_cams[0][:3]  # Shape [3, 224, 224]

            # Stack the 4th channels (CAMs) of all tensors along the channel dimension
            concatenated_cams = torch.stack([cam[3] for cam in all_cams], dim=0)  # Shape [n, 224, 224]

            # Combine the original RGB channels with the concatenated CAMs
            concatenated_cam_image = torch.cat((rgb_channels, concatenated_cams), dim=0)  # Shape [3 + n, 224, 224]

            # Asserting that the first 3 channels are the same as in the original RGB image
            for cam_image in all_cams:
                assert torch.equal(cam_image[:3], concatenated_cam_image[:3]), "The first 3 channels are not the same"
            
            # Saving concatenated CAM
            save_path = os.path.join(os.getcwd(), "res/augmented_images/hyper-kvasir/concatenated/", relative_file_path)
                                         
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(concatenated_cam_image, save_path)