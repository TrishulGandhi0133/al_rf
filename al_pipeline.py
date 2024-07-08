import torch
import torchvision
from ultralytics import YOLO
import numpy as np
import os
from PIL import Image
from roboflow import Roboflow
import contextlib

# Define margin sampling strategy
def margin_sampling(predictions, num_samples):
    margins = []
    for pred in predictions:
        confs = [d[4].item() for d in pred[0].boxes.data]  # Extract confidence scores from predictions
        if len(confs) < 2:
            margins.append(float('inf'))  # If less than 2 predictions, set margin to infinity
        else:
            top2_probs = np.sort(confs)[-2:]
            margin = top2_probs[1] - top2_probs[0]
            margins.append(margin)
    selected_indices = np.argsort(margins)[:num_samples]
    return selected_indices

# Define least confidence strategy
def least_confidence(predictions, num_samples):
    confidences = []
    for pred in predictions:
        if len(pred[0].boxes.data) == 0:
            confidences.append(float('inf'))  # If no predictions, set confidence to infinity
        else:
            confs = [d[4].item() for d in pred[0].boxes.data]
            confidences.append(np.max(confs))
    selected_indices = np.argsort(confidences)[:num_samples]
    return selected_indices
def active_learning_pipeline(api_key, workspace, project_name, version, num_unlabeled, strategy_name, model_name):
    # Download dataset from Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")
    
    # Map model name to the corresponding pretrained weights
    model_weights = {
        "YOLOv8n": "yolov8n-seg.pt",
        "YOLOv8s": "yolov8s-seg.pt",
        "YOLOv8m": "yolov8m-seg.pt",
        "YOLOv8l": "yolov8l-seg.pt",
        "YOLOv8x": "yolov8x-seg.pt"
    }
    
    if model_name not in model_weights:
        raise ValueError(f"Invalid model name. Choose from {list(model_weights.keys())}.")
    
    # Load YOLOv8 model with pretrained weights
    model = YOLO(model_weights[model_name])
    
    # Load labeled and unlabeled datasets
    labeled_path = os.path.join(dataset.location, "train/images")
    unlabeled_path = os.path.join(dataset.location, "test/images")
    
    labeled_dataset = [os.path.join(labeled_path, img) for img in os.listdir(labeled_path) if img.endswith(('jpg', 'jpeg', 'png'))]
    unlabeled_dataset = [os.path.join(unlabeled_path, img) for img in os.listdir(unlabeled_path) if img.endswith(('jpg', 'jpeg', 'png'))]
    #print(unlabeled_path)
    #print(labeled_dataset)
    
    print(f"Total unlabeled images: {len(unlabeled_dataset)}")
    print(f"Number of unlabeled images requested: {num_unlabeled}")
    
    # Run inference on the unlabeled dataset
    predictions = []
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            for img_path in unlabeled_dataset:
                img = Image.open(img_path)
                pred = model.predict(img, conf=0.25)  # Adjust confidence threshold as needed
                predictions.append(pred)

    # Select images based on the strategy
    if strategy_name == 'marginsampling':
        selected_indices = margin_sampling(predictions, num_unlabeled)
    elif strategy_name == 'leastconfidence':
        selected_indices = least_confidence(predictions, num_unlabeled)
    else:
        raise ValueError("Invalid strategy name. Choose either 'marginsampling' or 'leastconfidence'.")

    # Print the filenames of selected images
    selected_files = [unlabeled_dataset[idx] for idx in selected_indices]
    print("Selected images:", selected_files)

    return selected_files
