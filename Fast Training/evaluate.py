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

class COCODataset(CocoDetection):
    def __init__(self, root, dir_images, annFile, privileged_information_dirs=None, transforms=None, img_resize=None):
        super().__init__(os.path.join(root, dir_images), annFile)
        """
        Args:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            privileged_information_dirs (list): List of directories containing privileged information for each image.
            transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        """
        # Set image directory
        self.image_dir = os.path.join(root, dir_images)
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
                img, target_dict["boxes"], target_dict["labels"] = sample["image"], sample["bboxes"], sample["labels"]
            
            # Return image and target dictionary
            return img, target_dict
        
        except Exception as e:
            print(f"Error loading index {index}: {str(e)}")
            raise e
        
def main_function(
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
        BATCH_SIZE,
        NUM_WORKERS,
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

    # Creating model
    print("Creating model")

    model.transform.image_mean = channel_means
    model.transform.image_std = channel_stds
    model.transform.to(device)

    # Move the model to the correct device (e.g., CUDA or CPU)
    model = model.to(device)

    # Verify the model structure
    print(model)

    
    annotation_dir_test = os.path.join(DIR_TEST, DIR_ANNOTATIONS)

    test_dataset = COCODataset(root=DIR_TEST, annFile=annotation_dir_test, dir_images=DIR_IMAGES, privileged_information_dirs=PRIVILEGED_INFORMATION_DIRS, img_resize=IMG_RESIZE)#, get_valid_transform())

    # Define the batch size and number of workers for the DataLoader
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )

    print("Evaluating model")

    def apply_nms(pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
        """
        Applies non-maximum suppression (NMS) to the predicted boxes, scores, and labels.

        Parameters:
        - pred_boxes: Predicted bounding boxes (x_min, y_min, x_max, y_max).
        - pred_scores: Predicted scores for each bounding box.
        - pred_labels: Predicted class labels for each bounding box.

        Returns:
        - nms_boxes: Bounding boxes after NMS.
        - nms_scores: Scores after NMS.
        - nms_labels: Class labels after NMS.
        """
        if len(pred_boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        pred_boxes = torch.tensor(pred_boxes)
        pred_scores = torch.tensor(pred_scores)
        pred_labels = torch.tensor(pred_labels)

        keep_indices = ops.nms(pred_boxes, pred_scores, iou_threshold)

        return pred_boxes[keep_indices].numpy(), pred_scores[keep_indices].numpy(), pred_labels[keep_indices].numpy()

    """
    Evaluation Metrics:
    - Mean Average Precision (mAP)
    - Precision, Recall, and F1 Score (@IoU=0.5)
    - Euclidean Distance Error at different IoU thresholds
    """
    # Function to calculate the mean average precision (mAP) for the model 
    def calculate_map(model, test_data_loader, device, class_names, num_classes=NUM_CLASSES, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
        """
        Calculate the mean average precision (mAP) for the model on the validation set.

        Parameters:
        - model: The object detection model to evaluate.
        - test_data_loader: DataLoader for the test data.
        - device: Device ('cpu' or 'cuda') for computation.
        - class_names: List of class names for the dataset.
        - num_classes: Number of classes in the dataset.
        - iou_thresholds: List of IoU thresholds to use for mAP calculation.
        
        Returns:
        - ap_df: DataFrame containing AP for each class and threshold with mAP row.
        - pr_curves: Dictionary containing precision-recall coordinates for each class at each threshold.
        - all_pred_boxes, all_true_boxes, all_preds, all_labels, all_scores: List of predicted boxes, true boxes, predicted labels, true labels, and predicted scores for all images.
        """
        model.eval()
        all_results = []
        
        # Lists to store all predictions, true labels, and scores
        all_pred_boxes = []
        all_true_boxes = []
        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad(): # Iterate over the test data
            for images, targets in tqdm(test_data_loader, desc="Evaluating", unit=" batch"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Retrieving the model output for the given images
                output = model(images)

                # Iterate over the targets and outputs, and storing them in a result dictionary
                for i in range(len(targets)):
                    true_boxes = targets[i]["boxes"].cpu().numpy()
                    true_labels = targets[i]["labels"].cpu().numpy()

                    if i < len(output):
                        pred_boxes = output[i]["boxes"].cpu().numpy()
                        pred_labels = output[i]["labels"].cpu().numpy()
                        pred_scores = output[i]["scores"].cpu().numpy()

                        # Apply NMS here
                        pred_boxes, pred_scores, pred_labels = apply_nms(pred_boxes, pred_scores, pred_labels, iou_threshold=0.5)

                    else:
                        pred_boxes = np.empty((0, 4))
                        pred_labels = np.empty((0,))
                        pred_scores = np.empty((0,))

                    # Append the results to the respective lists
                    all_true_boxes.append(true_boxes)
                    all_pred_boxes.append(pred_boxes)
                    all_labels.append(true_labels)
                    all_preds.append(pred_labels)
                    all_scores.append(pred_scores)

                    result = {
                        "image_index": i,
                        "true_boxes": true_boxes,
                        "true_labels": true_labels,
                        "pred_boxes": pred_boxes,
                        "pred_labels": pred_labels,
                        "pred_scores": pred_scores,
                    }

                    all_results.append(result)

        # Creating dictionaries to store AP for each class and threshold, and precision-recall curves
        ap_per_class_per_threshold = {cls: {threshold: 0.0 for threshold in iou_thresholds} for cls in range(num_classes)}
        pr_curves = {cls: {threshold: {"precision": [], "recall": []} for threshold in iou_thresholds} for cls in range(num_classes)}

        # Iterating through the IoU thresholds
        for threshold in iou_thresholds:
            # Dictionary to store class metrics (TP, FP, FN) for each class
            class_metrics = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in range(num_classes)}

            # Iterating over all results
            for result in all_results:
                true_boxes = result["true_boxes"]
                true_labels = result["true_labels"]
                pred_boxes = result["pred_boxes"]
                pred_labels = result["pred_labels"]
                pred_scores = result["pred_scores"]

                # Filter predictions based on score threshold
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    # Computing the IoU Matrix to match predictions with true boxes
                    iou_matrix = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes)).numpy()

                    # Changing the IoU matrix to be a class-specific matrix
                    matched = np.zeros(len(true_boxes), dtype=bool)

                    # Assigning predictions to true boxes based on IoU threshold
                    for j in range(len(pred_boxes)):
                        max_iou = np.max(iou_matrix[j])
                        max_idx = np.argmax(iou_matrix[j])
                        pred_class = pred_labels[j]

                        # Check if the IoU is above the threshold and the prediction is correct (TP) or incorrect (FP)
                        if max_iou >= threshold:
                            if not matched[max_idx] and pred_class == true_labels[max_idx]:
                                class_metrics[pred_class]["TP"] += 1
                                matched[max_idx] = True
                            else:
                                class_metrics[pred_class]["FP"] += 1
                        else:
                            class_metrics[pred_class]["FP"] += 1

                    # Assigning false negatives (FN) for unmatched true boxes
                    for k, label in enumerate(true_labels):
                        if not matched[k]:
                            class_metrics[label]["FN"] += 1
                else:
                    for label in true_labels:
                        class_metrics[label]["FN"] += 1
                    for pred_class in pred_labels:
                        class_metrics[pred_class]["FP"] += 1

            # Computing precision and recall for each class
            for cls in range(num_classes):
                tp = class_metrics[cls]["TP"]
                fp = class_metrics[cls]["FP"]
                fn = class_metrics[cls]["FN"]

                # Calculating precision and recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                pr_curves[cls][threshold]["precision"].append(precision)
                pr_curves[cls][threshold]["recall"].append(recall)

                # Calculating the area under the precision-recall curve which is the AP
                ap = calculate_ap_from_pr_curve(pr_curves[cls][threshold]["precision"], pr_curves[cls][threshold]["recall"])
                ap_per_class_per_threshold[cls][threshold] = ap

        # Converting AP dictionary to DataFrame
        ap_df = pd.DataFrame.from_dict(ap_per_class_per_threshold, orient="index")
        ap_df.columns = [f"AP@{iou:.2f}" for iou in iou_thresholds]
        ap_df.index = [class_names[cls] for cls in range(num_classes)]

        # Add AP0.5:0.95 and mAP columns
        # Find the index of the column labeled "AP@0.50"
        start_idx = ap_df.columns.get_loc("AP@0.50")

        # Find the index of the column labeled "AP@0.95"
        end_idx = ap_df.columns.get_loc("AP@0.95") + 1  # +1 to include the last column

        # Compute AP@0.5:0.95 for each class (excluding the background class)
        ap_df["AP@0.5:0.95"] = ap_df.iloc[1:, start_idx:end_idx].mean(axis=1)

        # Adding mAP row
        # Using iloc to skip the first row (background)
        ap_df.loc["mAP"] = ap_df.iloc[1:].mean()

        print("\nAverage Precision Scores (AP) for each class at different IoU thresholds and mAP: ")
        print(ap_df)

        # Saving the AP DataFrame to a CSV file
        ap_df.to_csv(f'{SAVE_DIR}/average_precision.csv', index=True)

        # Saving the precision-recall curves to a json file
        with open(f'{SAVE_DIR}/pr_curves.json', 'w') as f:
            json.dump(pr_curves, f, indent=4)

        # based on the scores, ,ensure that all the lists are euqal by removing the extra elements
        all_pred_boxes = all_pred_boxes[:len(all_labels)]
        all_true_boxes = all_true_boxes[:len(all_labels)]
        all_preds = all_preds[:len(all_labels)]
        all_scores = all_scores[:len(all_labels)]

        return ap_df, pr_curves, all_pred_boxes, all_true_boxes, all_preds, all_labels, all_scores

    # Function to calculate the average precision (AP) from the precision-recall curve
    def calculate_ap_from_pr_curve(precision, recall):
        """
        Calculate the average precision (AP) from the precision-recall curve using the 11-point interpolation method.

        Parameters:
        - precision: List of precision values.
        - recall: List of recall values.

        Returns:
        - ap: Average precision (AP) calculated from the precision-recall curve.
        """
        precision = np.array(precision)
        recall = np.array(recall)
        
        # Ensure recall starts from 0 and ends at 1
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([1.0], precision, [0.0]))

        # Correct precision curve
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        # Compute AP as the area under the precision-recall curve
        indices = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        return ap

    # Function to calculate the precision, recall, and F1 score for the model
    def calculate_precision_recall_f1_all(all_pred_labels, all_true_labels, all_pred_boxes, all_true_boxes, class_names, iou_threshold=0.5):
        """
        Calculate precision, recall, and F1 score for object detection for all classes.

        Parameters:
        - all_pred_labels: List of predicted class labels for each image.
        - all_true_labels: List of true class labels for each image.
        - all_pred_boxes: List of predicted bounding boxes for each image.
        - all_true_boxes: List of true bounding boxes for each image.
        - class_names: List of class names.
        - iou_threshold: The threshold above which a prediction is considered correct.

        Returns:
        - class_metrics: DataFrame containing per-class precision, recall, F1 scores, and mean scores.
        """
        # Initialize class metrics dictionary
        class_metrics = {cls: {"TP": 0, "FP": 0, "FN": 0} for cls in range(len(class_names))}

        for pred_labels, true_labels, pred_boxes, true_boxes in zip(all_pred_labels, all_true_labels, all_pred_boxes, all_true_boxes):
            if len(pred_boxes) > 0 and len(true_boxes) > 0:
                # Computing the IoU Matrix to match predictions with true boxes (matching the predictions with the true boxes)
                iou_matrix = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes)).numpy()

                matched = np.zeros(len(true_boxes), dtype=bool)

                # Assigning predictions to true boxes based on the IoU threshold
                for j in range(len(pred_boxes)):
                    max_iou = np.max(iou_matrix[j])
                    max_idx = np.argmax(iou_matrix[j])
                    pred_class = pred_labels[j]

                    # Check if the IoU is above the threshold and the prediction is correct (TP) or incorrect (FP)
                    if max_iou >= iou_threshold:
                        if not matched[max_idx] and pred_class == true_labels[max_idx]:
                            class_metrics[pred_class]["TP"] += 1
                            matched[max_idx] = True
                        else:
                            class_metrics[pred_class]["FP"] += 1
                    else:
                        class_metrics[pred_class]["FP"] += 1

                # Assigning false negatives (FN) for unmatched true boxes
                for k, label in enumerate(true_labels):
                    if not matched[k]:
                        class_metrics[label]["FN"] += 1
            else:
                for label in true_labels:
                    class_metrics[label]["FN"] += 1
                for pred_class in pred_labels:
                    class_metrics[pred_class]["FP"] += 1

        # Compute per-class precision, recall, and F1
        metrics = []

        for cls in range(len(class_names)):
            tp = class_metrics[cls]["TP"]
            fp = class_metrics[cls]["FP"]
            fn = class_metrics[cls]["FN"]

            # Calculate precision, recall, and F1 score at the IoU threshold
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics.append([class_names[cls], precision, recall, f1])

        # Compute mean per-class metrics
        # Exclude the background class (first element)
        non_background_metrics = metrics[1:]

        mean_precision = np.mean([m[1] for m in non_background_metrics if m[1] > 0]) if any(m[1] > 0 for m in non_background_metrics) else 0
        mean_recall    = np.mean([m[2] for m in non_background_metrics if m[2] > 0]) if any(m[2] > 0 for m in non_background_metrics) else 0
        mean_f1        = np.mean([m[3] for m in non_background_metrics if m[3] > 0]) if any(m[3] > 0 for m in non_background_metrics) else 0


        # Create DataFrame for all metrics
        metrics.append(["Mean", mean_precision, mean_recall, mean_f1])
        class_metrics_df = pd.DataFrame(metrics, columns=["Class", "Precision", "Recall", "F1 Score"])

        # Save the results to a single CSV file
        class_metrics_df.to_csv(f'{SAVE_DIR}/precision_recall_f1_metrics.csv', index=False)

        # Print the results
        print("\nPer-Class Precision, Recall, and F1 Scores (including Mean) at IoU Threshold of {:.2f}:".format(iou_threshold))
        print(class_metrics_df)

        return class_metrics_df


    # Function the calculate the euclidean distance between the midpoints of predicted and true bounding boxes for small object detection
    def calculate_euclidean_distance_per_class(all_pred_boxes, all_true_boxes, all_pred_labels, all_scores, thresholds=np.arange(0.5, 1.0, 0.05), num_classes=NUM_CLASSES, class_names=None):
        """
        Calculate the Euclidean distance between midpoints of predicted and true bounding boxes,
        grouped by class and filtered by IoU threshold, considering the smallest distance for each prediction.

        Arguments:
        all_pred_boxes: numpy array of predicted boxes of shape (N, 4) in the format [xmin, ymin, xmax, ymax]
        all_true_boxes: numpy array of true boxes of shape (M, 4) in the format [xmin, ymin, xmax, ymax]
        all_pred_labels: numpy array of predicted class labels (length N)
        all_scores: numpy array of predicted scores (length N)
        thresholds: Array of IoU thresholds to filter predictions
        num_classes: Number of classes
        class_names: List of class names for display purposes

        Returns:
        df: pandas DataFrame containing Euclidean distances per class and mean distance for each threshold
        """
        # Initialize dictionaries to store distances for each class at each threshold
        euclidean_distances_per_class_at_threshold = {threshold: {i: [] for i in range(num_classes)} for threshold in thresholds}

        for pred_boxes, true_boxes, pred_labels, scores in zip(all_pred_boxes, all_true_boxes, all_pred_labels, all_scores):
            if len(pred_boxes) > 0 and len(true_boxes) > 0:
                # Compute IoU matrix to match predictions with true boxes
                iou_matrix = box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes)).numpy()
                
                # Calculate midpoints of predicted and true boxes (Euclidean distance between midpoints)
                pred_midpoints = [( (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ) for box in pred_boxes]
                true_midpoints = [( (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ) for box in true_boxes]

                # Iterate over predictions and calculate the Euclidean distance for each class and threshold
                for threshold in thresholds:
                    matched = np.zeros(len(true_boxes), dtype=bool)

                    # Iterate over predictions and calculate the Euclidean distance for each class and threshold
                    for j, pred_midpoint in enumerate(pred_midpoints):
                        # Find the maximum IoU and its index
                        max_iou = np.max(iou_matrix[j])
                        max_idx = np.argmax(iou_matrix[j])
                        pred_class = pred_labels[j]
                        score = scores[j]

                        # Check if the IoU is above the threshold and the prediction is correct
                        if max_iou >= threshold and not matched[max_idx]:
                            true_midpoint = true_midpoints[max_idx]
                            # Calculate Euclidean distance
                            distance = np.sqrt((pred_midpoint[0] - true_midpoint[0]) ** 2 + (pred_midpoint[1] - true_midpoint[1]) ** 2)
                            euclidean_distances_per_class_at_threshold[threshold][pred_class].append(distance)
                            matched[max_idx] = True

        # Prepare the results as a DataFrame
        data = {
            "Class": class_names if class_names else [f"Class {i}" for i in range(num_classes)]
        }

        # Calculate the mean Euclidean distance for each class and threshold but exclude the last class (background
        for threshold in thresholds:
            data[f"Mean Euclidean Distance@{threshold:.2f}"] = [
                np.mean(euclidean_distances_per_class_at_threshold[threshold][i]) if euclidean_distances_per_class_at_threshold[threshold][i] else np.nan
                for i in range(num_classes) # Exclude the last class (background)
            ]

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Save the results to a CSV file
        df.to_csv(f'{SAVE_DIR}/euclidean_distances.csv', index=True)

        # Print the DataFrame
        print("\nMean Euclidean Distances between Predicted and True Midpoints for each Class:")
        print(df)

        return df

    def filter_predictions(predictions, confidence_threshold=0.5, max_predictions=1000):
        """
        Filter predictions based on confidence threshold and maximum number of predictions.

        Args:
        - predictions (list): List of dictionaries containing 'boxes', 'scores', and 'labels' keys.
        - confidence_threshold (float): Minimum confidence score for predictions.
        - max_predictions (int): Maximum number of predictions to keep.

        """
        filtered = []
        for pred in predictions:
            # NMS
            mask = pred["scores"] > confidence_threshold
            filtered_boxes = pred["boxes"][mask]
            filtered_scores = pred["scores"][mask]
            filtered_labels = pred["labels"][mask]

            filtered.append({
                "boxes": filtered_boxes[:max_predictions],
                "scores": filtered_scores[:max_predictions],
                "labels": filtered_labels[:max_predictions],
            })
        return filtered

    def calculate_COCO_mAP(model, data_loader, device, iou_threshold=0.5):
        """
        Calculate the COCO mAP for the model using the provided data loader.

        Args:
        - model (torch.nn.Module): The object detection model to evaluate.
        - data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        - device (torch.device): Device for computation (CPU or CUDA).
        - iou_threshold (float): IoU threshold for mAP calculation.

        Returns:
        - mAP DataFrame: DataFrame containing mAP values for each class.    
        """
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)  # since we're using the corners method

        # Wrap the data_loader with tqdm for progress bar
        for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = model(images)
            filtered_predictions = filter_predictions(outputs, confidence_threshold=iou_threshold)
            metric.update(filtered_predictions, targets)

        # Computing results
        result = metric.compute()

        # Change the tensors into numpy arrays
        result = {key: (val.cpu().numpy() if isinstance(val, torch.Tensor) else val) for key, val in result.items()}

        # Ensure scalar results are in a list and provide proper index
        result = {key: val if isinstance(val, list) else [val] for key, val in result.items()}

        # Add all the results into a pandas DataFrame
        df = pd.DataFrame(result)

        print("\nCOCO mAP for each class:")
        print(df)

        # Save the results to a CSV file
        df.to_csv(f'{SAVE_DIR}/coco_mAP.csv', index=True)

        return df

    """
    Plotting Functions:
    - Plotting the Average Precision (AP) scores for each class at specific IoU thresholds (0.50 and 0.95)
    - Plotting the normalised confusion matrix
    """
    # Function to plot the Average Precision (AP) scores for each class at specific IoU thresholds (0.50 and 0.95)
    def plot_ap_bars(ap_df, iou_thresholds, class_names):
        """
        Plot bar chart of Average Precision (AP) scores for each class at specific IoU thresholds (0.50, 0.55, ..., 0.95).
        The values will be plotted on the bars, and a separate column for mAP will be included.

        Arguments:
        ap_df: pandas DataFrame containing AP scores per class and IoU thresholds.
        iou_thresholds: list of IoU thresholds (e.g., [0.50, 0.55, ..., 0.95])
        class_names: List of class names.
        """
        # Remove the first row for the background class
        ap_df = ap_df.iloc[1:].reset_index(drop=True)
        class_names = class_names[1:] + ['mAP']

        # Extract AP values for plotting
        ap_values = ap_df.iloc[:, :]
        ap_values.index = class_names

        # Remove the column names which are not in the IoU thresholds
        ap_values = ap_values[[col for col in ap_values.columns if str(col.split('@')[-1]) in iou_thresholds]]

        # Plot the AP scores for each class at the selected IoU thresholds
        ax = ap_values.plot(kind='bar', figsize=(14, 8), width=0.8, colormap='winter', edgecolor='black')

        # Customize the plot
        ax.set_xlabel("Classes", fontsize=12)
        ax.set_ylabel("Average Precision (AP)", fontsize=12)
        ax.set_title("Average Precision (AP) per Class at Various IoU Thresholds", fontsize=14)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylim([0, 1])

        ax.legend(title="IoU Thresholds", bbox_to_anchor=(1.05, 1), loc='upper left', title_fontsize=12, fontsize=10)

        # Add numeric values on the bars with 2 decimal places
        for container in ax.containers:
            ax.bar_label(container, labels=[f'{v:.2f}' for v in container.datavalues], label_type='edge', fontsize=10, color='black')

        # Tight layout to prevent clipping of labels
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{SAVE_DIR}/average_precision_bars.png', bbox_inches='tight', pad_inches=0)

        # Display the plot
        plt.show()

    # Function to plot the normalised confusion matrix
    def plot_normalised_confusion_matrix(all_pred_boxes, all_true_boxes, all_preds, all_labels, class_names, iou_threshold=0.5):
        """
        Plot the normalized confusion matrix for the model predictions using IoU mapping.
        In this version, background is represented by 0 (instead of -1), and background is included in the confusion matrix.
        
        Parameters:
        - all_pred_boxes: List of predicted bounding boxes for each image.
        - all_true_boxes: List of true bounding boxes for each image.
        - all_preds: List of predicted labels for each image.
        - all_labels: List of true labels for each image.
        - class_names: List of class names for the dataset (with the background class as the first element).
        - iou_threshold: IoU threshold for matching predictions to true boxes.
        """
        all_mapped_preds = []
        all_mapped_labels = []

        # Iterate through all images
        for pred_boxes, true_boxes, pred_labels, true_labels in zip(all_pred_boxes, all_true_boxes, all_preds, all_labels):
            # If there are no boxes, skip the image
            if len(pred_boxes) == 0 and len(true_boxes) == 0:
                continue

            # Compute IoU matrix between true boxes and predicted boxes
            iou_matrix = box_iou(torch.tensor(true_boxes), torch.tensor(pred_boxes)).numpy()

            # Keep track of which true boxes have been matched.
            # (Using -1 as an internal flag for "not yet matched")
            matched_true_boxes = np.full(len(true_boxes), -1)
            
            # Loop over predictions and match to true boxes (if possible)
            for pred_idx, pred_label in enumerate(pred_labels):
                # Skip if there are more predictions than true boxes in the IoU matrix
                if pred_idx >= iou_matrix.shape[0]:
                    continue

                max_iou = np.max(iou_matrix[pred_idx])
                max_idx = np.argmax(iou_matrix[pred_idx])

                if max_iou >= iou_threshold and max_idx < len(true_boxes) and matched_true_boxes[max_idx] == -1:
                    # Valid match: use the predicted label and the true label from the matched true box.
                    all_mapped_preds.append(pred_label)
                    all_mapped_labels.append(true_labels[max_idx])
                    matched_true_boxes[max_idx] = pred_label  # Mark this true box as matched.
                else:
                    # False positive: no matching true box; assign background (0) as the true label.
                    all_mapped_preds.append(pred_label)
                    all_mapped_labels.append(0)

            # Handle false negatives: unmatched true boxes.
            for true_idx, true_label in enumerate(true_labels):
                if matched_true_boxes[true_idx] == -1:
                    # No prediction matched: assign background (0) as the prediction.
                    all_mapped_preds.append(0)
                    all_mapped_labels.append(true_label)

        # Convert the collected mappings into numpy arrays.
        mapped_labels = np.array(all_mapped_labels)
        mapped_preds = np.array(all_mapped_preds)

        # Compute the confusion matrix using all classes (background included as index 0)
        cm = confusion_matrix(mapped_labels, mapped_preds, labels=range(len(class_names)))

        # Normalize the confusion matrix by row (i.e. for each true label)
        cm_normalized = cm.astype('float')
        row_sums = cm_normalized.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm_normalized / row_sums

        # Create the plot using seaborn heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm_normalized.T, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names, cbar=True, square=True)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Normalized Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/normalised_confusion_matrix.png', bbox_inches='tight', pad_inches=0)
        plt.show()


    # Example usage in the main evaluation function
    def evaluate(model, valid_data_loader, device, class_names):
        # Set the model to evaluation mode
        model.eval()

        # Calculate mAP and AP scores for each IoU threshold
        ap_df, pr_curves, all_pred_boxes, all_true_boxes, all_preds, all_labels, all_scores = calculate_map(model, valid_data_loader, device, class_names, num_classes=NUM_CLASSES, iou_thresholds=np.arange(0.0, 1.0, 0.05))
        
        calculate_COCO_mAP(model, valid_data_loader, device)

        # Calculate precision, recall, and F1 score for each class
        calculate_precision_recall_f1_all(all_preds, all_labels, all_pred_boxes, all_true_boxes, class_names, iou_threshold=0.5)

        # Calculate the Euclidean distance between midpoints of predicted and true bounding boxes (Check viability)
        calculate_euclidean_distance_per_class(all_pred_boxes, all_true_boxes, all_preds, all_scores, num_classes=NUM_CLASSES, class_names=class_names)
        
        # Plot the AP bar graph for each class
        plot_ap_bars(ap_df, iou_thresholds=["0.50", "0.75", "0.90", "0.5:0.95"], class_names=class_names)

        # Plot the normalized confusion matrix
        # plot_normalised_confusion_matrix(all_pred_boxes, all_true_boxes, all_preds, all_labels, class_names)

    # Load the best model weights
    model.load_state_dict(torch.load(f'{SAVE_DIR}/weights/best.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Retrieving the class names
    class_names = [cls["name"] for cls in CLASSES]

    # Evaluate the model on the test set
    evaluate(model, test_data_loader, device, class_names=class_names)

    # Reset data loader iteration
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )

    def plot_batches_as_mosaic(test_data_loader, device, num_batches=3, channel_indices=[0, 1, 2], class_definitions=CLASSES, resize_size=(1920, 1080), iou_threshold=0.5):
        """
        Plots three figures, one for each batch, showing images as a 2x2 grid with bounding boxes and class labels.
        
        Parameters:
        - test_data_loader: DataLoader for the test data.
        - device: The device (CPU or GPU) for computation.
        - num_batches: Number of batches to plot.
        - channel_indices: List of channel indices to extract for visualisation (e.g., [0, 1, 2] for RGB).
        - class_definitions: List of class dictionaries containing id, name, and color.
        - resize_size: Size to resize the images for display.
        """
        class_lookup = {cls["id"]: (cls["name"], tuple(cls["color"])) for cls in class_definitions}

        # Evaluate the model on the test set
        model.eval()

        # Create a single iterator for the data loader
        data_iter = iter(test_data_loader)

        for batch_idx in range(num_batches):
            try:
                # Load a batch of data from the iterator
                images, targets = next(data_iter)
            except StopIteration:
                print("No more batches available in test_data_loader!")
                break

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            with torch.no_grad():
                outputs = model(images)

            # Filter predictions based on score threshold (e.g., 0.5)
            for i in range(len(outputs)):
                keep = outputs[i]['scores'] >= iou_threshold
                outputs[i] = {k: v[keep] for k, v in outputs[i].items()}

            # Create a 2x2 grid for visualization of ground truth
            num_images = len(images)
            grid_size = math.ceil(np.sqrt(num_images))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(23, 15))
            axes = axes.flatten()

            for img_idx, (image, target) in enumerate(zip(images, targets)):
                sample_image = image.cpu().numpy()

                # Keep only the specified channels
                selected_channels = [sample_image[i] for i in channel_indices]
                selected_channels = [(channel * 255).astype(np.uint8) for channel in selected_channels]
                sample_image_rgb = np.stack(selected_channels, axis=-1).astype(np.uint8)

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

                for i, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_id = labels[i]
                    class_name, color = class_lookup.get(class_id, (f'Class {class_id}', (255, 255, 255)))

                    cv2.rectangle(sample_image_rgb, (x_min, y_min), (x_max, y_max), color, 10)

                    font_scale = 2
                    (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)
                    padding = 10
                    text_width += 2 * padding
                    text_height += 2 * padding

                    cv2.rectangle(sample_image_rgb, (x_min, y_min - text_height - 5),
                                (x_min + text_width, y_min - 5), color, -1)
                    cv2.putText(sample_image_rgb, class_name, (x_min + padding, y_min - padding - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

                sample_image_rgb = cv2.resize(sample_image_rgb, resize_size)
                axes[img_idx].imshow(sample_image_rgb)
                axes[img_idx].axis('off')

            for ax in axes[num_images:]:
                ax.axis('off')

            fig.suptitle(f"Test Batch {batch_idx + 1} Ground Truth", fontsize=20, weight='bold', color='white', y=0.92)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.9, bottom=0.1, left=0.1, right=0.9)
            fig.patch.set_facecolor('xkcd:dark blue')
            plt.savefig(f'{SAVE_DIR}/test_batch_{batch_idx + 1}_labels.png', bbox_inches='tight', pad_inches=0)
            plt.show()

            # Now plot the predictions
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(23, 15))
            axes = axes.flatten()

            for img_idx, (image, target) in enumerate(zip(images, targets)):
                sample_image = image.cpu().numpy()
                selected_channels = [sample_image[i] for i in channel_indices]
                selected_channels = [(channel * 255).astype(np.uint8) for channel in selected_channels]
                sample_image_rgb = np.stack(selected_channels, axis=-1).astype(np.uint8)

                boxes = outputs[img_idx]['boxes'].cpu().detach().numpy()
                labels = outputs[img_idx]['labels'].cpu().detach().numpy()
                scores = outputs[img_idx]['scores'].cpu().detach().numpy()
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

                for i, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_id = labels[i]
                    class_name, color = class_lookup.get(class_id, (f'Class {class_id}', (255, 255, 255)))
                    class_name = f"{class_name} {scores[i]:.0%}"

                    cv2.rectangle(sample_image_rgb, (x_min, y_min), (x_max, y_max), color, 10)

                    font_scale = 2
                    (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)
                    padding = 10
                    text_width += 2 * padding
                    text_height += 2 * padding

                    cv2.rectangle(sample_image_rgb, (x_min, y_min - text_height - 5),
                                (x_min + text_width, y_min - 5), color, -1)
                    cv2.putText(sample_image_rgb, class_name, (x_min + padding, y_min - padding - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

                sample_image_rgb = cv2.resize(sample_image_rgb, resize_size)
                axes[img_idx].imshow(sample_image_rgb)
                axes[img_idx].axis('off')

            for ax in axes[num_images:]:
                ax.axis('off')

            fig.suptitle(f"Test Batch {batch_idx + 1} Predictions (IoU={iou_threshold})", fontsize=20, weight='bold', color='white', y=0.92)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.9, bottom=0.1, left=0.1, right=0.9)
            fig.patch.set_facecolor('xkcd:dark blue')
            plt.savefig(f'{SAVE_DIR}/test_batch_{batch_idx + 1}_predictions.png', bbox_inches='tight', pad_inches=0)
            plt.show()

    # Example usage
    plot_batches_as_mosaic(test_data_loader, device, num_batches=3, channel_indices=[0, 1, 2], class_definitions=CLASSES)



# Get the arguments from argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast Training for Object Detection')
    parser.add_argument('--DIR_INPUT', type=str, help='Input Directory')
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
    parser.add_argument('--BATCH_SIZE', type=int, help='Batch Size')
    parser.add_argument('--NUM_WORKERS', type=int, help='Number of Workers')
    parser.add_argument('--model_name', type=str, help='Model Name')
    args = parser.parse_args()

    # Run the main function
    main_function(
        args.DIR_INPUT,
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
        args.BATCH_SIZE,
        args.NUM_WORKERS,
        args.model_name
    )


    