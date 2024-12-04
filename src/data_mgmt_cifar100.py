# Imported modules
from config import params

import os
import pickle
import csv
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def save_cifar100_in_directory(data_root):
    """
    Processes the CIFAR-100 dataset:
        - Downloads the dataset if not already present
        - Organizes images into directories based on coarse and fine labels
        - Saves the images in '.jpg' format
        - Creates a CSV file with details of the images
    
    Args:
        data_root (str): The root directory where the CIFAR-100 dataset will be stored and processed
    """
    
    def load_coarse_labels(file):
        """
        Loads coarse labels from CIFAR-100 dataset files
        
        Args:
            file (str): Path to the CIFAR-100 dataset file
        
        Returns:
            list: List of coarse labels
        """
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            return data['coarse_labels']

    def save_images(dataset, directory, coarse_labels, set_name, csv_writer):
        """
        Saves images from the CIFAR-100 dataset into organized directories and writes details to CSV
        
        Args:
            dataset (Dataset): The CIFAR-100 dataset
            directory (str): The root directory where images will be saved
            coarse_labels (list): List of coarse labels corresponding to the dataset
            set_name (str): The name of the set ('train' or 'test')
            csv_writer (csv.writer): The CSV writer object for logging image details
        """
        
        print(f"Saving images to {directory}")
        for idx, (img, fine_label) in tqdm(enumerate(dataset)):
        
            # Get the coarse label for the current image
            coarse_label = coarse_labels[idx]

            # Get the class names
            fine_class_name = fine_label_names[fine_label]
            coarse_class_name = coarse_label_names[coarse_label]
        
            # Create a subdirectory for each superclass and class
            class_dir = os.path.join(directory, coarse_class_name, fine_class_name)
            os.makedirs(class_dir, exist_ok=True)
        
            # Save the image
            img_name = f'{idx}.jpg'
            img_path = os.path.join(class_dir, img_name)
            img.save(img_path, 'JPEG')

            # Write to CSV
            csv_writer.writerow([set_name, coarse_class_name, fine_class_name, img_name])

    # Load the CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True)
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=True)

    # Load meta file to get class names
    with open(os.path.join(train_dataset.root, 'cifar-100-python', 'meta'), 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
        fine_label_names = meta['fine_label_names']
        coarse_label_names = meta['coarse_label_names']

    # Load coarse labels for training and test sets
    train_coarse_labels = load_coarse_labels(os.path.join(train_dataset.root, 'cifar-100-python', 'train'))
    test_coarse_labels = load_coarse_labels(os.path.join(train_dataset.root, 'cifar-100-python', 'test'))

    # Create directories for storing images
    train_dir = os.path.join(data_root, "cifar-100-python/cifar-100_train")
    test_dir = os.path.join(data_root, "cifar-100-python/cifar-100_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Open CSV file for writing
    csv_file_path = os.path.join(data_root, 'cifar-100-python/image_labels.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Set (train or test)', 'Coarse Label', 'Fine Label', 'File Name'])
    
        # Save training images and write to CSV
        save_images(train_dataset, train_dir, train_coarse_labels, 'train', csv_writer)
    
        # Save test images and write to CSV
        save_images(test_dataset, test_dir, test_coarse_labels, 'test', csv_writer)

    print("Images have been saved and CSV file has been created successfully.")


# Dictionary for enumerating fine labels (0-indexed)
LABEL_TO_INDEX = {
    "beaver": 0,
    "dolphin": 1,
    "otter": 2,
    "seal": 3,
    "whale": 4,
    "aquarium_fish": 5,
    "flatfish": 6,
    "ray": 7,
    "shark": 8,
    "trout": 9,
    "orchid": 10,
    "poppy": 11,
    "rose": 12,
    "sunflower": 13,
    "tulip": 14,
    "bottle": 15,
    "bowl": 16,
    "can": 17,
    "cup": 18,
    "plate": 19,
    "apple": 20,
    "mushroom": 21,
    "orange": 22,
    "pear": 23,
    "sweet_pepper": 24,
    "clock": 25,
    "keyboard": 26,
    "lamp": 27,
    "telephone": 28,
    "television": 29,
    "bed": 30,
    "chair": 31,
    "couch": 32,
    "table": 33,
    "wardrobe": 34,
    "bee": 35,
    "beetle": 36,
    "butterfly": 37,
    "caterpillar": 38,
    "cockroach": 39,
    "bear": 40,
    "leopard": 41,
    "lion": 42,
    "tiger": 43,
    "wolf": 44,
    "bridge": 45,
    "castle": 46,
    "house": 47,
    "road": 48,
    "skyscraper": 49,
    "cloud": 50,
    "forest": 51,
    "mountain": 52,
    "plain": 53,
    "sea": 54,
    "camel": 55,
    "cattle": 56,
    "chimpanzee": 57,
    "elephant": 58,
    "kangaroo": 59,
    "fox": 60,
    "porcupine": 61,
    "possum": 62,
    "raccoon": 63,
    "skunk": 64,
    "crab": 65,
    "lobster": 66,
    "snail": 67,
    "spider": 68,
    "worm": 69,
    "baby": 70,
    "boy": 71,
    "girl": 72,
    "man": 73,
    "woman": 74,
    "crocodile": 75,
    "dinosaur": 76,
    "lizard": 77,
    "snake": 78,
    "turtle": 79,
    "hamster": 80,
    "mouse": 81,
    "rabbit": 82,
    "shrew": 83,
    "squirrel": 84,
    "maple_tree": 85,
    "oak_tree": 86,
    "palm_tree": 87,
    "pine_tree": 88,
    "willow_tree": 89,
    "bicycle": 90,
    "bus": 91,
    "motorcycle": 92,
    "pickup_truck": 93,
    "train": 94,
    "lawn_mower": 95,
    "rocket": 96,
    "streetcar": 97,
    "tank": 98,
    "tractor": 99
}


# Data transforms to be used for training and evaluation of models
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    "train_aug": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, *[np.mean([0.485, 0.456, 0.406]).item()] * 1],
            std=[0.229, 0.224, 0.225, *[np.mean([0.485, 0.456, 0.406]).item()] * 1]
        )
    ]),
    
    "eval": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    "eval_aug": transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, *[np.mean([0.485, 0.456, 0.406]).item()] * 1],
            std=[0.229, 0.224, 0.225, *[np.mean([0.485, 0.456, 0.406]).item()] * 1]
        )
    ])
}


class CIFAR100Dataset(Dataset):
    """
    Custom PyTorch Dataset object for the CIFAR-100 dataset.

    Attributes:
        paths (list): Stores file paths to each image in the dataset.
        labels (list): Stores the numerical label (an integer corresponding to a class) of each image in the dataset. 
        data_transforms (string): Specifies whether data transforms should be for training or evaluation.
        augmented_data (bool): Indicates whether the dataset consists of standard images or augmented images with explanations.
    """

    def __init__(self, paths, labels, transform_type, augmented_data):
        """
        Initializes the HyperKvasirDataset instance.

        Args:
            paths (list): List of file paths to the images.
            labels (list): List of numerical labels corresponding to each image.
            transform_type (str): Specifies whether data transforms should be for training ('train') or evaluation ('eval').
            augmented_data (bool): Indicates whether the dataset consists of augmented images.
        """

        self.paths = paths
        self.labels = labels

        assert transform_type == "train" or transform_type == "eval"
        if augmented_data:
            transform_type += "_aug"
        self.data_transforms = data_transforms[transform_type]

        self.augmented_data = augmented_data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """

        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and its label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image tensor, its label, the file path, and the index.
        """

        path = self.paths[idx]
        label = self.labels[idx]
        
        # Load image from image or tensor file depending on whether data is augmented or not
        if self.augmented_data:
            image = torch.load(path)
        else:
            image = Image.open(path)
        
        image = self.data_transforms(image)
               
        return image, label, path, idx


def prepare_data_cifar100(seed, augmented_data, model_explanation, split, batch_size=params["batch_size"]):
    """
    Processes the CIFAR-100 dataset to return PyTorch Dataset and DataLoader objects.

    Args:
        seed (int): Seed for random number generation to ensure reproducibility.
        augmented_data (bool): Indicates whether the dataset consists of augmented images with explanations.
        model_explanation (str): Name of the model providing explanations, e.g., 'densenet161'.
        split (bool): If True, splits Datasets and DataLoader objects into training, validation, and test sets. 
                      If False, returns one Dataset and one DataLoader object containing all images from CIFAR-100.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        If `split` is True:
            Tuple[CIFAR100Dataset, CIFAR100Dataset, CIFAR100Dataset, DataLoader, DataLoader, DataLoader]:
            train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader
        
        If `split` is False:
            Tuple[CIFAR100Dataset, DataLoader]: complete_dataset, complete_dataloader
    """

    # Setting seed as specified
    if seed:
        torch.manual_seed(seed=seed)

    # Storing paths to images and corresponding labels
    if augmented_data:
        img_dir_path = os.path.join(os.getcwd(), f"res/augmented_images/cifar-100-python/{model_explanation}")
        csv_file_path = os.path.join(os.getcwd(), "res/augmented_images/cifar-100-python")
    else:
        img_dir_path = os.path.join(os.getcwd(), "res/cifar-100-python")

    # File handling for HyperKvasir data
    paths = []
    labels = []

    if augmented_data:
        file = open(os.path.join(csv_file_path, "image-labels.csv"), "r")
    else:
        file = open(os.path.join(img_dir_path, "image-labels.csv"), "r")
    file.readline()

    for line in file:

        set_type, coarse_label, fine_label, file_name = line.strip().split(",")

        file_name += ".pt" if augmented_data else ".jpg"
        set_type = f"cifar-100_{set_type}"

        path_to_img_file = os.path.join(
            img_dir_path, set_type, coarse_label, fine_label, file_name
        )

        enumerated_label = LABEL_TO_INDEX[fine_label]

        # Add image and label to stored paths and labels
        paths.append(path_to_img_file)
        labels.append(enumerated_label)

    file.close()

    # If not splitting data in sets, Dataset and DataLoader objects are created and returned containing all images in the dataset (used for creating CAMs)
    if not split:

        complete_dataset = CIFAR100Dataset(paths, labels, transform_type="eval", augmented_data=False)
        complete_dataloader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False)
        
        return complete_dataset, complete_dataloader

    # Performing split of dataset

    # First splitting test set and the rest
    temp_paths, test_paths, temp_labels, test_labels = train_test_split(paths, labels, test_size=params["test_size"], stratify=labels if params["stratify"] else None, random_state=seed)

    # Then splitting training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(temp_paths, temp_labels, test_size=params["val_size"], stratify=temp_labels if params["stratify"] else None, random_state=seed)

    # Asserting that sets are disjoint
    assert set(train_paths).isdisjoint(set(val_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(val_paths).isdisjoint(set(test_paths))

    # Creating Datasets and DataLoaders
    train_dataset = CIFAR100Dataset(train_paths, train_labels, transform_type="train", augmented_data=augmented_data)
    val_dataset = CIFAR100Dataset(val_paths, val_labels, transform_type="eval", augmented_data=augmented_data)
    test_dataset = CIFAR100Dataset(test_paths, test_labels, transform_type="eval", augmented_data=augmented_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    # Downloading and storing CIFAR-100 dataset when running this file
    data_root = os.path.join(os.getcwd(), "res")
    save_cifar100_in_directory(data_root)