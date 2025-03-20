import pandas as pd
import numpy as np
import cv2
import random
import os
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import warnings
import json
from collections import defaultdict
warnings.filterwarnings('ignore')
import argparse
# from PIL import Image
# import re

# # For models and training
import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
# # https://pytorch.org/vision/0.20/_modules/torchvision/models/detection/retinanet.html
# # https://pytorch.org/vision/main/models.html

# # For data loading
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
from albumentations.pytorch import ToTensorV2

# # For evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
from torchvision.ops import box_iou
import torchvision.ops as ops
from sklearn.preprocessing import label_binarize, LabelEncoder
import seaborn as sns
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial

# Seeding everything for reproducibility
seed = 42 #7 # The Perfect Number #12 # The number of Apostles 
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  # For multi-GPU  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  # Some set this to True
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)# raise error if CUDA >= 10.2
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)

def collate_fn(batch):
    return tuple(zip(*batch))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

DIR_IMAGES = 'images'

class COCODataset(CocoDetection):
    def __init__(self, root, dir_images, annFile, privileged_information_dirs=None, transforms=None, img_resize=None):
        super().__init__(os.path.join(root, DIR_IMAGES), annFile)
        """
        Args:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            privileged_information_dirs (list): List of directories containing privileged information for each image.
            transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """
        # Set image directory
        self.image_dir = os.path.join(root, DIR_IMAGES)
        # Setting the privileged information directories
        if privileged_information_dirs is not None:
            self.privileged_dirs = [os.path.join(root, p) for p in privileged_information_dirs] if privileged_information_dirs else None
        else:
            self.privileged_dirs = None

        # Set transforms
        self.transforms = transforms
        self.img_resize = img_resize

    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, index):
        try:
            # Retrieve the image and target
            img, target = super().__getitem__(index)
            # Storing the original size of the image
            original_size = img.size
            
            # Transform image (resize, convert to tensor, normalize)
            if self.img_resize:
                img = img.resize(self.img_resize)

            img = F.to_tensor(img)
            # Normalize image by dividing by 255 (it is being scaled already from 0-255 to 0-1, through super().__getitem__)
            # img /= 255.0
            
            # Cloning for student network
            rgb_img = img.clone()
            
            # Load privileged information if available
            if self.privileged_dirs:
                # Load the images from the privileged directories
                privileged = [cv2.imread(os.path.join(priv_dir, self.coco.loadImgs(self.ids[index])[0]["file_name"]), cv2.IMREAD_UNCHANGED) for priv_dir in self.privileged_dirs]
                # Normalize privileged information
                privileged = [(p / 255.0).astype(np.float32) for p in privileged] # Normalize privileged information
                # Convert to tensor and add channel dimension
                privileged_tensors = [torch.tensor(p).unsqueeze(0) for p in privileged] # Convert to tensor and add channel dimension
                # Apply resize to the privileged information
                if self.img_resize:
                    privileged_tensors = [F.to_tensor(cv2.resize(p, self.img_resize)) for p in privileged]

                # Concatenate image and privileged information
                img = torch.cat([img] + privileged_tensors, dim=0)
            
            # Error Chacking that the whole tensor is between 0 and 1
            if torch.any(img < 0) or torch.any(img > 1):
                raise ValueError("Image tensor is not between 0 and 1")

            # Process annotations
            boxes = []
            labels = []
            sizes = []
            for annotation in target:
                # Load bounding box coordinates
                bbox = annotation["bbox"]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Convert to (x_min, y_min, x_max, y_max (Pascol VOC format))
                # Resize bounding box coordinates
                if self.img_resize:
                    bbox = [bbox[0] / original_size[0] * self.img_resize[0], bbox[1] / original_size[1] * self.img_resize[1], bbox[2] / original_size[0] * self.img_resize[0], bbox[3] / original_size[1] * self.img_resize[1]]

                # Append bounding box and label
                boxes.append(bbox)
                if annotation["category_id"] != 0: # Skip background class
                    labels.append(annotation["category_id"])
                    # labels.append(1)

                sizes.append(original_size)
            
            # Convert to tensors
            target_dict = {
                "boxes": torch.FloatTensor(boxes),
                "labels": torch.LongTensor(labels),
                "image_id": torch.tensor([index]),
                "size": torch.tensor(sizes)
            }

            # Ensure that targer boxes are of shape (N, 4)
            if target_dict["boxes"].shape[0] == 0:
                target_dict["boxes"] = torch.zeros((0, 4))
            elif len(target_dict["boxes"].shape) == 1:
                target_dict["boxes"] = target_dict["boxes"].unsqueeze(0)
            
            # Apply transforms (Data Augmentation)
            if self.transforms:
                sample = {"image": img, "bboxes": target_dict["boxes"], "labels": target_dict["labels"]}
                sample = self.transforms(sample)
                rgb_img, img, target_dict["boxes"], target_dict["labels"] = sample["image"], sample["bboxes"], sample["labels"]
            
            # Return image and target dictionary
            return rgb_img, img, target_dict
        
        except Exception as e:
            print(f"Error loading index {index}: {str(e)}")
            raise e
        
def main_function(
        DIR_INPUT,
        DIR_TRAIN,
        DIR_VALID,
        DIR_TEST,
        DIR_IMAGES,
        DIR_ANNOTATIONS,
        IMG_RESIZE,
        SAVE_DIR,
        CLASSES,
        NUM_CLASSES,
        PRIVILEGED_INFORMATION_DIRS,
        NUM_CHANNELS,
        img_means,
        img_stds,
        ALL_PRIVILEGED_INFORMATION_DIRS,
        model,
        teacher_model,
        ALPHA,
        BATCH_SIZE,
        NUM_WORKERS,
        student_hook_layer,
        teacher_hook_layer,
        model_name,
    ):
    
    # Device for pytorch
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Creating output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(f'{SAVE_DIR}/weights', exist_ok=True)

    # Print the number of channels
    print(f"Number of input channels: {NUM_CHANNELS}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Saving to: {SAVE_DIR}")

    # Retrieve the correct indexes based on which channels are being used
    def get_channel_indexes(privileged_information_dirs):
        indexes = []
        for priv_dir in privileged_information_dirs:
            index = ALL_PRIVILEGED_INFORMATION_DIRS.index(priv_dir)
            indexes.append(index)
        # Add 3 for all the indexes since the first 3 channels are RGB
        indexes = [i + 3 for i in indexes]
        return indexes

    # Get the indexes of the privileged information channels
    privileged_indexes = get_channel_indexes(PRIVILEGED_INFORMATION_DIRS)

    # Add the RGB channels to the privileged indexes
    privileged_indexes = [0, 1, 2] + privileged_indexes

    channel_means = [img_means[i] for i in privileged_indexes]
    channel_stds = [img_stds[i] for i in privileged_indexes]

    rgb_means = [img_means[0], img_means[1], img_means[2]]
    rgb_stds = [img_stds[0], img_stds[1], img_stds[2]]
    
    # Creating model
    print("Creating model")

    model.transform.image_mean = rgb_means
    model.transform.image_std = rgb_stds
    model.transform.to(device)

    # Move the model to the correct device (e.g., CUDA or CPU)
    model = model.to(device)

    # Verify the model structure
    print(model)

    # Creating teacher model
    print("Creating teacher model")

    teacher_model.transform.image_mean = channel_means
    teacher_model.transform.image_std = channel_stds
    teacher_model.transform.to(device)

    # Move the model to the correct device (e.g., CUDA or CPU)
    teacher_model = teacher_model.to(device)

    # Setting teacher model to evaluation mode
    teacher_model.eval()

    # Verify the model structure
    print(teacher_model)

    # Functions to be used for training and validation and for DataLoader
    class Averager:
        def __init__(self):
            self.current_total = 0.0
            self.iterations = 0.0

        def send(self, value):
            self.current_total += value
            self.iterations += 1

        @property
        def value(self):
            if self.iterations == 0:
                return 0
            else:
                return 1.0 * self.current_total / self.iterations

        def reset(self):
            self.current_total = 0.0
            self.iterations = 0.0

    """
    Creating the Datasets and DataLoaders
    """
    annotation_dir_train = os.path.join(DIR_TRAIN, DIR_ANNOTATIONS)
    annotation_dir_valid = os.path.join(DIR_VALID, DIR_ANNOTATIONS)
    annotation_dir_test = os.path.join(DIR_TEST, DIR_ANNOTATIONS)

    train_dataset = COCODataset(root=DIR_TRAIN, annFile=annotation_dir_train, dir_images=DIR_IMAGES, privileged_information_dirs=PRIVILEGED_INFORMATION_DIRS, img_resize=IMG_RESIZE)#, get_train_transform())
    valid_dataset = COCODataset(root=DIR_VALID, annFile=annotation_dir_valid, dir_images=DIR_IMAGES, privileged_information_dirs=PRIVILEGED_INFORMATION_DIRS, img_resize=IMG_RESIZE)#, get_valid_transform())
    # test_dataset = COCODataset(root=DIR_TEST, annFile=annotation_dir_test, dir_images=DIR_IMAGES, privileged_information_dirs=PRIVILEGED_INFORMATION_DIRS, img_resize=IMG_RESIZE)#, get_valid_transform())

    # Define the batch size and number of workers for the DataLoader
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, # True
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False, # True
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )

    # test_data_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=collate_fn,
    #     worker_init_fn=seed_worker,
    #     generator=g
    # )

    # Plotting the first batch of the train_data_loader
    def plot_batches_as_mosaic(train_data_loader, device, num_batches=3, channel_indices=[0, 1, 2], class_definitions=CLASSES, resize_size=(1920, 1080)):
        """
        Plots three figures, one for each batch, showing images as a 2x2 grid with bounding boxes and class labels.
        
        Parameters:
        - train_data_loader: Data loader for training data.
        - device: The device (CPU or GPU) for computation.
        - num_batches: Number of batches to plot.
        - channel_indices: List of channel indices to extract for visualisation (e.g., [0, 1, 2] for RGB).
        - class_definitions: List of class dictionaries containing id, name, and color.
        - resize_size: Size to resize the images for display.
        """
        class_lookup = {cls["id"]: (cls["name"], tuple(cls["color"])) for cls in class_definitions}

        data_iter = iter(train_data_loader)

        for batch_idx in range(num_batches):
            # Load a batch of data
            try:
                # Load a batch of data from the iterator
                images, priv_images, targets = next(data_iter)
            except StopIteration:
                print("No more batches available in test_data_loader!")
                break

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Create a 2x2 grid for visualization
            num_images = len(images)
            grid_size = math.ceil(np.sqrt(num_images))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(23, 15))

            # Flatten axes for easy iteration
            axes = axes.flatten()

            for img_idx, (image, target) in enumerate(zip(images, targets)):
                # Extract and reshape the image
                sample_image = image.cpu().numpy()

                # Keep only the specified channels
                selected_channels = [sample_image[i] for i in channel_indices]
                
                # Convert channels to uint8 format
                selected_channels = [(channel * 255).astype(np.uint8) for channel in selected_channels]

                # Stack the selected channels to form the image
                sample_image_rgb = np.stack(selected_channels, axis=-1)

                # Convert to uint8
                sample_image_rgb = sample_image_rgb.astype(np.uint8)

                # Get bounding boxes and labels
                boxes = target['boxes'].cpu().numpy()
                labels = target['labels'].cpu().numpy()
                size = target['size'].cpu().numpy()

                # Resize the image and bounding boxes based on the original size
                original_size_x, original_size_y = size[0]
                resize_ratio_x = original_size_x / IMG_RESIZE[0]
                resize_ratio_y = original_size_y / IMG_RESIZE[1]

                # resizing the bounding boxes
                for box in boxes:
                    box[0] *= resize_ratio_x
                    box[2] *= resize_ratio_x
                    box[1] *= resize_ratio_y
                    box[3] *= resize_ratio_y

                sample_image_rgb = cv2.resize(sample_image_rgb, size[0])           

                # Draw bounding boxes and labels
                for i, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = map(int, box)

                    # Get the class name and color for the current box
                    class_id = labels[i]
                    class_name, color = class_lookup.get(class_id, (f'Class {class_id}', (255, 255, 255)))

                    # Draw the rectangle (bounding box) on the image
                    cv2.rectangle(sample_image_rgb, (x_min, y_min), (x_max, y_max), color, 10)

                    # Calculate the size of the text
                    font_scale = 2
                    (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)

                    # Add some padding around the text for the background
                    padding = 10
                    text_width += 2 * padding  # Add padding to the left and right
                    text_height += 2 * padding  # Add padding to the top and bottom

                    # Draw the background rectangle for the class label text
                    cv2.rectangle(sample_image_rgb, (x_min, y_min - text_height - 5),
                                (x_min + text_width, y_min - 5), color, -1)

                    # Put the class name text (adjusted for padding)
                    cv2.putText(sample_image_rgb, class_name, (x_min + padding, y_min - padding - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

                # Resize the image for display 2k resolution
                sample_image_rgb = cv2.resize(sample_image_rgb, resize_size)

                # Plot the image
                axes[img_idx].imshow(sample_image_rgb)
                axes[img_idx].axis('off')

            # Hide any unused axes
            for ax in axes[num_images:]:
                ax.axis('off')

            fig.suptitle(f"Train Batch {batch_idx + 1}", fontsize=20, weight='bold', color='white', y=0.92)
            plt.tight_layout()
            
            # Adjust space between subplots to make sure all are same size
            plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.9, bottom=0.1, left=0.1, right=0.9)

            # Make the figure background gray
            fig.patch.set_facecolor('xkcd:dark blue')
            # Save the figure
            plt.savefig(f'{SAVE_DIR}/train_batch_{batch_idx + 1}.png', bbox_inches='tight', pad_inches=0)

            plt.show()

    # Example usage
    plot_batches_as_mosaic(train_data_loader, device, num_batches=3, channel_indices=[0, 1, 2], class_definitions=CLASSES)


    # Extract trainable parameters (those requiring gradients)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Length of trainable parameters: ", len(params))
    # Display all the model parameters which are trainable
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    print("\nTrainable parameters: ")
    for name in names:
        print(name)

    # Optimizer: AdamW is effective for object detection tasks
    # 'lr' is the learning rate (had previously tried 0.001, 0.0001, 0.00001)
    # Also tried SGD with momentum but AdamW was more faster and converged to the same optimal value
    optimizer = torch.optim.AdamW(params, lr=0.0001)

    # Learning rate scheduler: Reduces the learning rate when validation loss plateaus
    # 'patience' determines how many epochs to wait before reducing the learning rate
    # 'factor' is the rate of decay (0.5 halves the learning rate)
    lr_scheduler = None
    # torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', patience=2, factor=0.5, verbose=True
    # )

    # Early stopping: Stops training after a set number of epochs without improvement
    # 'patience' is the number of epochs to wait before stopping
    patience = 5

    # Number of epochs for training
    NUM_EPOCHS = 100

    student_features_dict = {}
    teacher_features_dict = {}

    def student_hook_fn(module, input, output):
        student_features_dict["features"] = output  # Save student features

    def teacher_hook_fn(module, input, output):
        teacher_features_dict["features"] = output  # Save teacher features

    # Register hooks
    """
    In RetinaNet, model.backbone.body.layer4[2] is the last layer of the ResNet backbone.
    This is where we will extract features from the student and teacher models.
    The next layer will be the FPN (Feature Pyramid Network)
    """
    student_hook = student_hook_layer.register_forward_hook(student_hook_fn)
    teacher_hook = teacher_hook_layer.register_forward_hook(teacher_hook_fn)

    # For student training with the involvement of the teacher
    def cosine_distance_loss(student_features, teacher_features):
        """
        Compute cosine distance loss between student and teacher features.

        Args:
            student_features (torch.Tensor): Feature map from the student model (B, 2048, H, W).
            teacher_features (torch.Tensor): Feature map from the teacher model (B, 2048, H, W).

        Returns:
            torch.Tensor: Scalar cosine distance loss.
        """
        # Global Average Pooling to reduce (B, 2048, H, W) -> (B, 2048)
        """
            Getting the features for all the batches, and removing height and width, since they are all the same
        """
        student_pooled = torch.nn.functional.adaptive_avg_pool2d(student_features, (1, 1)).squeeze(-1).squeeze(-1)
        teacher_pooled = torch.nn.functional.adaptive_avg_pool2d(teacher_features, (1, 1)).squeeze(-1).squeeze(-1)

        # Normalize feature vectors
        """
            Normalizing the features to have a magnitude of 1, to make the cosine similarity calculation easier
        """
        student_pooled = torch.nn.functional.normalize(student_pooled, p=2, dim=1)  # (B, 2048)
        teacher_pooled = torch.nn.functional.normalize(teacher_pooled, p=2, dim=1)  # (B, 2048)

        # Compute cosine similarity
        """
            Cosine Similarity Loss = (|A.B| / (|A| * |B|))
            Cosine Distance Loss = 1 - Cosine Similarity Loss

            where A and B are the feature vectors
        """
        cosine_sim = torch.nn.functional.cosine_similarity(student_pooled, teacher_pooled, dim=1)  # (B,)

        # Convert similarity to proper cosine distance
        cosine_dist = 1 - cosine_sim  # (B,)

        return cosine_dist.mean()  # Scalar loss
    


    print("Starting training...")

    # Helper function to plot losses
    def plot_training_summary_with_df(loss_df):
        """
        Plots the training and validation loss over epochs, with beautified colors and appearance.
        
        Parameters:
        - loss_df: DataFrame containing columns for 'train_loss' and 'val_loss'.
        """
        # Define custom colors
        train_color = '#1f3a8a'  # Dark Blue
        val_color = '#28a745'    # Green

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot total train and validation losses
        plt.plot(loss_df.index, loss_df['train_loss'], label='Training Loss', color=train_color, linewidth=2)
        plt.plot(loss_df.index, loss_df['val_loss'], label='Validation Loss', color=val_color, linewidth=2)

        # Customize plot
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Losses', fontsize=16)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/training_summary.png', bbox_inches='tight', pad_inches=0)

        # Show plot
        plt.show()

    # Helper function to plot individual losses
    def plot_individual_losses(loss_df):
        """
        Plots individual loss values (like train_loss, validation_loss, etc.) on separate subplots in a dynamic layout.
        The layout has len(columns)//2 columns.
        
        Parameters:
        - loss_df: DataFrame containing columns for various loss values (e.g., 'train_loss', 'val_loss', etc.).
        """
        # Extract all the columns except epoch
        loss_columns = [col for col in loss_df.columns if col != 'epoch']
        num_plots = len(loss_columns)
        
        # Determine the number of columns (half the number of loss columns)
        num_cols = num_plots // 2
        num_rows = 2 if num_plots > num_cols else 1  # 1 or 2 rows depending on the number of columns
        
        # Create subplots with dynamic layout (half the number of loss columns as the number of columns)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6), sharey=True)
        
        # If only one row, make axes a 1D array for easier indexing
        if num_rows == 1:
            axes = np.array([axes])  # Convert to 2D array for consistent indexing
        
        # Loop through the loss columns and plot each on a separate subplot
        for idx, loss_col in enumerate(loss_columns):
            # If num_rows is 1, axes is 1D, so we access it with just idx
            if num_rows == 1:
                ax = axes[idx]
            else:
                # Otherwise, we access the subplot in the grid as a 2D array
                ax = axes[idx // num_cols, idx % num_cols]
            
            # Plot original loss
            ax.plot(loss_df.index, loss_df[loss_col], color='darkblue', linewidth=2)
            
            # Set labels and title
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{loss_col.replace("_", " ").title()} Over Epochs', fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/individual_losses.png', bbox_inches='tight', pad_inches=0)
        plt.show()


    # Training and validation loop
    def train_and_validate(model, train_data_loader, test_data_loader, optimizer=optimizer, lr_scheduler=lr_scheduler, num_epochs=NUM_EPOCHS, patience=patience, device=device):
        """
        Train and validate the model for a specified number of epochs, with early stopping and learning rate scheduling.
        
        Parameters:
        - model: The object detection model to train.
        - train_data_loader: DataLoader for the training data.
        - test_data_loader: DataLoader for the validation data.
        - optimizer: Optimizer used for training.
        - lr_scheduler: Learning rate scheduler (optional).
        - num_epochs: Number of epochs to train the model.
        - patience: Number of epochs to wait before early stopping.
        - device: Device ('cpu' or 'cuda') for training.
        
        Returns:
        - loss_dict_values: Dictionary with loss values for each epoch.
        - epoch_losses: List of average losses for each epoch.
        - val_epoch_losses: List of average validation losses for each epoch.
        - val_dict_values: Dictionary with validation losses for each epoch.
        - loss_df: DataFrame containing all loss values for better tracking.
        """
        # Initialize tracking variables
        patience_counter = 0
        best_loss = float('inf')
        epoch_losses = []
        val_epoch_losses = []
        loss_dict_values = []
        val_dict_values = []
        time_elapsed_per_epoch = []

        model.to(device)
        
        # Loop over epochs
        for epoch in range(num_epochs):
            model.train()
            loss_values = []  # Collect batch losses for epoch averaging
            avg_loss_dict = {}  # Average the loss dictionary values
            start_time = time.time()  # Measure epoch time

            # Training loop
            total_iterations = len(train_data_loader)
            with tqdm(total=total_iterations, unit=" batch", desc=f"Epoch #{epoch+1}/{num_epochs} (Train)") as tepoch:
                for batch, (images, priv_image, targets) in enumerate(train_data_loader, 1):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = model(images, targets)

                    # Update the Average loss dictionary
                    if not avg_loss_dict:
                        avg_loss_dict = {k: [v.item()] for k, v in loss_dict.items()}
                    else:
                        avg_loss_dict = {k: v + [loss_dict[k].item()] for k, v in avg_loss_dict.items()}

                    # Privileged information
                    priv_images = list(priv.to(device) for priv in priv_image)
                    teacher_outputs = teacher_model(priv_images)

                    # Retrieve features from the student and teacher models
                    student_features = student_features_dict["features"]
                    teacher_features = teacher_features_dict["features"]

                    # Compute cosine distance loss
                    cosine_loss = cosine_distance_loss(student_features, teacher_features)

                    # Total (Student) loss (classification + bounding box regression)
                    losses = sum(loss for loss in loss_dict.values())

                    # For knowledge distillation, add cosine distance loss to the total loss
                    losses = (1 - ALPHA) * losses + ALPHA * cosine_loss

                    # Track total loss
                    loss_value = losses.item()
                    loss_values.append(loss_value)  # Track batch loss

                    # Backward pass and optimizer step
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    if model_name != "Faster R-CNN":
                        # Update tqdm description and increment progress
                        tepoch.set_postfix(
                            batch=batch, 
                            batch_loss=loss_value, 
                            loss_cls=loss_dict['classification'].item(), 
                            loss_box=loss_dict['bbox_regression'].item(),
                        )
                    else:
                        tepoch.set_postfix(
                            batch=batch, 
                            batch_loss=loss_value, 
                            loss_cls=loss_dict['loss_classifier'].item(), 
                            loss_box=loss_dict['loss_box_reg'].item(),
                            loss_obj=loss_dict['loss_objectness'].item(),
                            loss_rpn=loss_dict['loss_rpn_box_reg'].item()
                        )
                    tepoch.update(1)

            # Average the loss dictionary values for the epoch
            avg_loss_dict = {k: np.mean(v) for k, v in avg_loss_dict.items()}
            loss_dict_values.append(avg_loss_dict)

            # Scheduler step (if provided)
            if lr_scheduler:
                lr_scheduler.step(np.mean(loss_values))

            # Average epoch loss and logging
            epoch_loss_value = sum(loss_values) / len(train_data_loader)
            epoch_losses.append(epoch_loss_value)

            # Measure epoch time
            time_elapsed = time.time() - start_time
            time_elapsed_per_epoch.append(time_elapsed)
            tepoch.set_postfix(
                epoch_loss=epoch_loss_value,
                epoch_time=f"{time_elapsed // 60:.0f}m.{time_elapsed % 60:.0f}s"
            )

            # Save model weights after each epoch
            torch.save(model.state_dict(), f'{SAVE_DIR}/weights/last.pth')

            # Validation loop
            val_loss_values = []
            with torch.no_grad():
                with tqdm(total=len(test_data_loader), unit=" batch", desc=f"Epoch #{epoch+1}/{num_epochs} (Validation)") as vepoch:
                    for images, _, targets in test_data_loader:
                        images = list(image.to(device) for image in images)
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                        # Forward pass
                        loss_dict = model(images, targets)
                        val_loss = sum(loss for loss in loss_dict.values()).item()
                        val_loss_values.append(val_loss)

                        vepoch.set_postfix(batch_loss=val_loss)
                        vepoch.update(1)

            # Average validation loss
            val_epoch_loss = sum(val_loss_values) / len(test_data_loader)
            val_epoch_losses.append(val_epoch_loss)

            # Store validation loss dictionary values for tracking
            val_dict_values.append({k: np.mean([v.item() for v in loss_dict.values()]) for k in loss_dict.keys()})

            # Early stopping condition
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                patience_counter = 0

                # Save the best model weights
                torch.save(model.state_dict(), f'{SAVE_DIR}/weights/best.pth')
            else:
                patience_counter += 1

            print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss_value:.4f}, Validation Loss: {val_epoch_loss:.4f}, Patience Counter: {patience_counter}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Convert loss dictionary values to a DataFrame for tracking
        loss_dict_values = {k: [d[k] for d in loss_dict_values] for k in loss_dict_values[0].keys()}
        val_dict_values = {k: [d[k] for d in val_dict_values] for k in val_dict_values[0].keys()}

        # Create a DataFrame from the collected loss values
        loss_df = pd.DataFrame({
            'epoch': np.arange(1, len(epoch_losses) + 1),
            'train_loss': epoch_losses,
            'val_loss': val_epoch_losses
        })

        for key in loss_dict_values:
            loss_df[f'{key}_train'] = loss_dict_values[key]

        for key in val_dict_values:
            loss_df[f'{key}_val'] = val_dict_values[key]

        # Save the results to a CSV file
        loss_df.to_csv(f'{SAVE_DIR}/results.csv', index=True)

        # Save the time taken per epoch
        time_elapsed_df = pd.DataFrame({
            'epoch': np.arange(1, len(time_elapsed_per_epoch) + 1),
            'time_elapsed': time_elapsed_per_epoch,
            'total_time_elapsed': np.cumsum(time_elapsed_per_epoch),
        })
        time_elapsed_df.to_csv(f'{SAVE_DIR}/time_elapsed.csv', index=True)

        # Plot the training summary
        plot_training_summary_with_df(loss_df)
        plot_individual_losses(loss_df)

        # Remove hooks
        student_hook.remove()
        teacher_hook.remove()

        # Return results
        return loss_dict_values, epoch_losses, val_epoch_losses, val_dict_values, loss_df

    # Train the model
    loss_dict_values, epoch_losses, val_epoch_losses, val_dict_values, loss_df = train_and_validate(
        model, train_data_loader, valid_data_loader)


# Get the arguments from argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast Training for Object Detection')
    parser.add_argument('--DIR_INPUT', type=str, help='Input Directory')
    parser.add_argument('--DIR_TRAIN', type=str, help='Training Directory')
    parser.add_argument('--DIR_VALID', type=str, help='Validation Directory')
    parser.add_argument('--DIR_TEST', type=str, help='Testing Directory')
    parser.add_argument('--DIR_IMAGES', type=str, help='Images Directory')
    parser.add_argument('--DIR_ANNOTATIONS', type=str, help='Annotations Directory')
    parser.add_argument('--IMG_RESIZE', type=tuple, help='Image Resize Tuple')
    parser.add_argument('--SAVE_DIR', type=str, help='Save Directory')
    parser.add_argument('--CLASSES', type=list, help='Classes List')
    parser.add_argument('--NUM_CLASSES', type=int, help='Number of Classes')
    parser.add_argument('--PRIVILEGED_INFORMATION_DIRS', type=list, help='Privileged Information Directories')
    parser.add_argument('--NUM_CHANNELS', type=int, help='Number of Channels')
    parser.add_argument('--img_means', type=list, help='Image Means')
    parser.add_argument('--img_stds', type=list, help='Image Standard Deviations')
    parser.add_argument('--ALL_PRIVILEGED_INFORMATION_DIRS', type=list, help='All Privileged Information Directories')
    parser.add_argument('--model', type=torch.nn.Module, help='Model')
    parser.add_argument('--teacher_model', type=torch.nn.Module, help='Teacher Model')
    parser.add_argument('--ALPHA', type=float, help='Alpha')
    parser.add_argument('--BATCH_SIZE', type=int, help='Batch Size')
    parser.add_argument('--NUM_WORKERS', type=int, help='Number of Workers')
    parser.add_argument('--student_hook_layer', type=partial, help='Student Hook')
    parser.add_argument('--teacher_hook_layer', type=partial, help='Teacher Hook')
    parser.add_argument('--model_name', type=str, help='Model Name')
    args = parser.parse_args()

    # Run the main function
    main_function(
        args.DIR_INPUT,
        args.DIR_TRAIN,
        args.DIR_VALID,
        args.DIR_TEST,
        args.DIR_IMAGES,
        args.DIR_ANNOTATIONS,
        args.IMG_RESIZE,
        args.SAVE_DIR,
        args.CLASSES,
        args.NUM_CLASSES,
        args.PRIVILEGED_INFORMATION_DIRS,
        args.NUM_CHANNELS,
        args.img_means,
        args.img_stds,
        args.ALL_PRIVILEGED_INFORMATION_DIRS,
        args.model,
        args.teacher_model,
        args.ALPHA,
        args.BATCH_SIZE,
        args.NUM_WORKERS,
        args.student_hook_layer,
        args.teacher_hook_layer,
        args.model_name
    )


    