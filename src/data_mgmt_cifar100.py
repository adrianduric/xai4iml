import os
import pickle
import csv
from torchvision import datasets
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
    csv_file_path = os.path.join(data_root, 'cifar-100-python/cifar100_dataset.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Set (train or test)', 'Coarse Label', 'Fine Label', 'File Name'])
    
        # Save training images and write to CSV
        save_images(train_dataset, train_dir, train_coarse_labels, 'train', csv_writer)
    
        # Save test images and write to CSV
        save_images(test_dataset, test_dir, test_coarse_labels, 'test', csv_writer)

    print("Images have been saved and CSV file has been created successfully.")


# Specify the root directory where the dataset will be processed
data_root = os.path.join(os.getcwd(), "res")
save_cifar100_in_directory(data_root)