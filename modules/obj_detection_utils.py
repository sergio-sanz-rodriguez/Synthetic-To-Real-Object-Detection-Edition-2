"""
Provides utility functions for deep learning object detection workflows in PyTorch.  
Some functions are based on https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py"
"""

import datetime
import errno
import os
import time
import cv2
import torch
import random
import hashlib
import numbers
import numpy as np
import torch.distributed as dist
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List
from collections import defaultdict, deque
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F
from torchvision.transforms import v2 as T


def collate_fn(batch):
    return tuple(zip(*batch))

# Function to set random seed
def set_seeds(seed: int=42):
    """Sets random seeds for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


# Function to remove redundant boxes and masks
def prune_predictions(
    pred,
    score_threshold=0.66,
    iou_threshold=0.5,
    best_candidate="area",
    remove_large_boxes=None
    ):

    """
    Filters and refines predictions by:
    1. Removing unusually large bounding boxes if specified
    2. Removing low-confidence detections based on the score threshold.
    3. Applying a binary mask threshold to filter out weak segmentation masks.
    4. Using Non-Maximum Suppression (NMS) to eliminate overlapping predictions.
    6. Ensuring the highest-scoring prediction is always included.
    7. Selecting the best-confident bounding box based on a criterion: largest area or highest score.

    Args:
        pred: The raw predictions containing "boxes", "scores", "labels", and "masks".
        score_threshold: The minimum confidence score required to keep a prediction (default: 0.66).
        iou_threshold: The Intersection over Union (IoU) threshold for NMS (default: 0.5).
        best_candidate: Selects, from the final set of bounding boxes, the best one based on a criterion:
            -"area": the bounding box with the largest area is chosen
            -"score": the bounding boxe with the highest score is chosen
            -None: no criteion is used, maining the pruning method may contain one or more best bounding box candidates

    Returns:
        A dictionary with filtered and refined predictions:
            "boxes": Tensor of kept bounding boxes.
            "scores": Tensor of kept scores.
            "labels": Tensor of kept labels.
    """
    
    # Validate score_threshold
    if not isinstance(score_threshold, (int, float)) or not (0 <= score_threshold <= 1):
        raise ValueError("'score_threshold' must be a float between 0 and 1")

    # Validate iou_threshold
    if not isinstance(iou_threshold, (int, float)) or not (0 <= iou_threshold <= 1):
        raise ValueError("'iou_threshold' must be a float between 0 and 1")

    # Validate best_candidate
    if best_candidate not in ("area", "score", None):
        raise ValueError("'best_candidate' must be one of: 'area', 'score', or None")
    
    # Validate remove_large_boxes
    if remove_large_boxes is not None and not isinstance(remove_large_boxes, numbers.Number):
        raise ValueError("'remove_large_boxes' must be a numeric value or None")

    # Filter big boxes
    if remove_large_boxes is not None:
        areas = (pred["boxes"][:, 2] - pred["boxes"][:, 0]) * (pred["boxes"][:, 3] - pred["boxes"][:, 1])
        keep_idx = areas < remove_large_boxes
        pred["boxes"] = pred["boxes"][keep_idx]
        pred["scores"] = pred["scores"][keep_idx]
        pred["labels"] = pred["labels"][keep_idx]


    # Filter predictions based on confidence score threshold
    scores = pred["scores"]

    if len(scores) == 0:
        return {
            "boxes": [],
            "scores": [],
            "labels": []
            }

    best_idx = scores.argmax()
    high_conf_idx = scores > score_threshold

    # Extract the best bounding box, score, and label
    best_pred = {
        "boxes": pred["boxes"][best_idx].unsqueeze(0).long(), 
        "scores": pred["scores"][best_idx].unsqueeze(0),
        "labels": pred["labels"][best_idx].unsqueeze(0),
    }

    filtered_pred = {
        "boxes":  pred["boxes"][high_conf_idx].long(),
        "scores": pred["scores"][high_conf_idx],
        "labels": pred["labels"][high_conf_idx], #[f"roi: {s:.3f}" for s in scores[high_conf_idx]]
    }

    # Apply Non-Maximum Suppression (NMS) to remove overlapping predictions
    if len(filtered_pred["boxes"]) == 0:
        if len(best_pred["boxes"]) > 0:
            return best_pred
        else:
            return filtered_pred 
    
    keep_idx = ops.nms(filtered_pred["boxes"].float(), filtered_pred["scores"], iou_threshold)

    # Return filtered predictions
    keep_preds = {
        "boxes": filtered_pred["boxes"][keep_idx],
        "scores": filtered_pred["scores"][keep_idx],
        "labels": filtered_pred["labels"][keep_idx], #[i] for i in keep_idx],
    }

    # Ensure the best prediction is always included
    best_box = best_pred["boxes"][0]
    if not any(torch.equal(best_box, box) for box in keep_preds["boxes"]):
        keep_preds["boxes"] = torch.cat([keep_preds["boxes"], best_pred["boxes"]])
        keep_preds["scores"] = torch.cat([keep_preds["scores"], best_pred["scores"]])
        keep_preds["labels"] = torch.cat([keep_preds["labels"], best_pred["labels"]])
    
    # Now we have a set of good candidates. Let's take the best one based on a criterion
    if keep_preds["boxes"].shape[0] > 1:

        # Return only the one with the highest score
        if best_candidate == "score":            
            idx = keep_preds['scores'].argmax().item()
            #_, idx = keep_preds['scores'].topk(2)
            final_pred = {
                "boxes": keep_preds["boxes"][idx].unsqueeze(0),
                "scores": keep_preds["scores"][idx].unsqueeze(0),
                "labels": keep_preds["labels"][idx].unsqueeze(0),
            }
            return final_pred

        # Compute area of each box and return the one with the largest area
        elif best_candidate == "area":
            areas = (keep_preds["boxes"][:, 2] - keep_preds["boxes"][:, 0]) * (keep_preds["boxes"][:, 3] - keep_preds["boxes"][:, 1])
            idx = areas.argmax().item()
            #_, idx = keep_preds['scores'].topk(2)  
            final_pred = {
                "boxes": keep_preds["boxes"][idx].unsqueeze(0),
                "scores": keep_preds["scores"][idx].unsqueeze(0),
                "labels": keep_preds["labels"][idx].unsqueeze(0),
            }
            return final_pred
        
    return keep_preds
       


# Function to display images with masks and boxes on the ROIs
def display_and_save_predictions(
    preds: List=None,
    dataloader: torch.utils.data.Dataset | torch.utils.data.DataLoader = None,
    box_color: str='white',
    mask_color: str='blue',
    width: int=1,
    font_type: str=None,
    font_size: int=8,
    print_classes: bool=True,
    print_scores: bool=True,
    label_to_class_dict={1: 'roi'},
    save_dir: str = None
    ):

    """
    This function displays images with predicted bounding boxes and segmentation masks.
    Arguments:
        preds (List): A list of predictions, each containing 'boxes', 'labels', 'scores', and optionally 'masks'.
        dataloader (torch.utils.data.DataLoader): A DataLoader object containing the images.
        box_color (str): Color of the bounding boxes drawn on the image.
        mask_color (str): Color of the segmentation masks drawn on the image.
        width (int): The width of the bounding box lines.
        print_classes (bool): If True, the labels will be printed on the bounding boxes.
        print_scores (bool): If True, the confidence scores will be printed on the bounding boxes.
        label_to_class_dict (dict): Dictionary mapping label indices to class names.
        save_dir (str, optional): Path to save images. If None, images will not be saved.
    """

    plt.close("all")

    # Convert dataset to DataLoader if needed
    if isinstance(dataloader, torch.utils.data.Dataset):
        dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Create save directory if saving images
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    # Number of images
    num_images = len(preds)
    cols = 3 
    rows = (num_images + cols - 1) // cols

    # Set up the grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # Loop through the predictions and process each image
    for idx, (data, filtered_pred) in enumerate(zip(dataloader.dataset, preds)):  

        # Get the image from the dataset
        image, _ = data
        
        # Prepare the image
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)

        # Taking the first 3 channels if it's RGB
        image = image[:3, ...]  

        # Replace labels with strings for proper display
        labels = [
            f"{label_to_class_dict[l.item()] if print_classes else ''}{': ' + f'{s.item():.3f}' if print_scores else ''}".strip(": ")
            for l, s in zip(filtered_pred["labels"], filtered_pred["scores"])
        ]

        # Draw bounding boxes
        if len(filtered_pred["boxes"]) > 0:
            output_image = draw_bounding_boxes(
                image=image,
                boxes=filtered_pred["boxes"],
                labels=labels if print_classes or print_scores else None,
                colors=box_color,
                width=width,
                font=font_type,
                font_size=font_size)
        else:
            output_image = image

        # Save Image (if save_dir is provided)
        if save_dir:
            image_pil = to_pil_image(output_image)  # Convert tensor to PIL Image
            image_path = os.path.join(save_dir, f"prediction_{idx+1}.png")
            image_pil.save(image_path)

        # Plot on the grid
        ax = axes[idx]
        ax.imshow(output_image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.set_title(f"Prediction {idx + 1}")
        ax.axis("off")

    # Hide unused subplots (if any)
    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_transformed_data(img, target, transformed_img, transformed_target):

    """
    Visualizes the original and transformed image along with bounding boxes and masks.
    
    Parameters:
    - img: Original image tensor.
    - target: Original target dictionary (contains boxes, masks, labels).
    - transformed_img: Transformed image tensor.
    - transformed_target: Transformed target dictionary.
    """
    
    # Visualize original image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Original Image
    axes[0].imshow(img.permute(1, 2, 0))  # Convert CHW to HWC for plotting
    for box in target['boxes']:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[0].add_patch(rect)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Transformed Image
    axes[1].imshow(transformed_img.permute(1, 2, 0))  # Convert CHW to HWC for plotting
    for box in transformed_target['boxes']:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='g', facecolor='none'
        )
        axes[1].add_patch(rect)
    axes[1].set_title('Transformed Image')
    axes[1].axis('off')

    plt.show()

class RandomCircleOcclusion(T.RandomErasing):

    """
    Applies random circular occlusions to an image tensor. Inherits from torchvision.transforms.RandomErasing.

    This transform overlays multiple randomly placed circles of various sizes and predefined colors onto 
    the image. It is useful for data augmentation to simulate occlusions or visual noise.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), num_elems=6):

        """
        Initializes the RandomCircleOcclusion class.

        Args:
            p (float, optional): Probability of applying the occlusion. Defaults to 0.5.
            scale (tuple, optional): Range of scale factors for the occlusion. Defaults to (0.02, 0.2).
            ratio (tuple, optional): Range of aspect ratios for the occlusion. Defaults to (0.3, 3.3).

        Note:
            The occlusion color is set to a forest green color.
        """

        # Define the forest green color in a tuple format (R, G, B)
        forest_green = (34/255, 139/255, 34/255)  # Normalized RGB values
        self.colors = [
            (0, 0, 0),          # Black
            (53, 94, 59),       # Dark Green
            (211, 211, 211),    # Light Gray
            (34, 139, 34),      # Forest Green
        ]
        
        super().__init__(p=p, scale=scale, ratio=ratio, value=forest_green)
        self.num_elems = num_elems

    def forward(self, img, target=None):

        """
        Applies the random circular occlusion to the input image tensor.

        Args:
            img (Tensor): Image tensor with shape (C, H, W).
            target (optional): Target corresponding to the image, returned unchanged.

        Returns:
            Tuple[Tensor, Any]: Transformed image and original target.
        """

        if torch.rand(1).item() > self.p:
            return img, target

        img_np = img.mul(255).byte().permute(1, 2, 0).numpy()  # Convert tensor to uint8 numpy
        h, w, _ = img_np.shape

        # Generate a random occlusion mask
        num_blobs = np.random.randint(2, self.num_elems)
        for _ in range(num_blobs):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            size = np.random.randint(h // 20, h // 5)
            #scale_sample = np.random.uniform(self.scale[0], self.scale[1])
            #size = int(np.sqrt(scale_sample * h * w / np.pi))
            selected_color = np.array(self.colors[np.random.randint(len(self.colors))], dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), size, 255, -1)

            # Apply selected color
            img_np[mask > 0] = selected_color

        img_tensor = torch.from_numpy(img_np).float().div(255).permute(2, 0, 1)  # Convert back to tensor
        return img_tensor, target


class RandomTextureOcclusion:

    """
    Applies random occlusions to an image using RGBA texture objects (e.g., plants) 
    to simulate natural occlusions. Occlusions are applied stochastically with a 
    user-defined probability.

    This class is useful for data augmentation in computer vision tasks such as object 
    detection or classification, where robustness to occlusion is desired.
    """

    def __init__(self, obj_path, scale=(0.2, 0.5), transparency=0.5, p=0.5):

        """
        Initializes the RandomTextureOcclusion class.

        Args:
            obj_path (list): List of paths to plant images.
            scale (tuple, optional): Range of scale factors for the occlusion. Defaults to (0.2, 0.5).
            transparency (float, optional): Transparency of the plant image. Defaults to 0.5.
            p (float, optional): Probability of applying the occlusion. Defaults to 0.5.
        """

        #obj_path = ["T_Bush_Falcon.png"]
        obj_images = [Image.open(path).convert("RGBA") for path in obj_path]
        self.scale = scale
        self.obj_images = obj_images 
        self.transparency = transparency
        self.p = p

    def __call__(self, img, target=None):

        """
        Applies a randomly placed, scaled, and rotated texture occlusion to the input image.

        Args:
            img (PIL.Image or torch.Tensor): Input image.
            target: Optional target data (e.g., labels or bounding boxes).

        Returns:
            Tuple: Transformed image as torch.Tensor, and the unmodified target.
        """

        if random.random() > self.p or not self.obj_images:
            return img, target

        # Convert to PIL Image if necessary
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Select a random plant image        
        obj_img = random.choice(self.obj_images)
        
        # Resize plant to a random size
        w, h = img.size       
        scale = random.uniform(self.scale[0], self.scale[1])  # Random scale factor
        new_h, new_w = int(w * scale), int(w * scale)
        new_h, new_w = min(new_w, w), min(new_h, h)
        obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random rotation
        angle = random.uniform(0, 360)
        obj_img = obj_img.rotate(angle, expand=True)
        # Make sure it fits into base image (w, h)
        new_w, new_h = obj_img.size
        if new_w >= w or new_h >= h:
            scale = min(w / new_w, h / new_h) * 0.8  # a bit smaller than max
            new_w, new_h = int(new_w * scale), int(new_h * scale)
            obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random placement
        x_offset = random.randint(0, w - new_w)
        y_offset = random.randint(0, h - new_h)        

        # Convert plant image to numpy for blending
        obj_np = np.array(obj_img).astype(float)
        img_np = np.array(img).astype(float)

        # Alpha channel
        obj_alpha = obj_np[:, :, 3] > 128
        
        # Apply transparency
        # Blend the RGB channels with the alpha channel
        for c in range(3):  # Iterate over RGB channels
            img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                obj_alpha * obj_np[:, :, c] + (1 - obj_alpha) * img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )

        # Convert back to PIL
        img_occluded = Image.fromarray(img_np.astype(np.uint8))
        
        return F.pil_to_tensor(img_occluded), target

import hashlib

class RandomTextureOcclusionDeterministic:

    """
    Applies deterministic occlusions using RGBA texture objects (e.g., plants) based on 
    the hash of the input image or its filename. This ensures reproducibility of the 
    augmentation, which is useful for validation or consistent visual testing.
    """

    def __init__(self, obj_path, scale=(0.2, 0.5), transparency=0.5, p=0.5):
        
        """
        Initializes the RandomTextureOcclusionDeterministic class.

        Args:
            obj_path (list): List of paths to RGBA plant images to be used for occlusion.
            scale (tuple, optional): Range of scale factors for the occlusion. Defaults to (0.2, 0.5).
            transparency (float): Transparency level for occlusions (not currently used).
            p (float): Probability of applying the occlusion.
        """

        self.obj_images = [Image.open(path).convert("RGBA") for path in obj_path]
        self.scale = scale
        self.transparency = transparency
        self.p = p

    def __call__(self, img, target=None):

        """
        Applies a reproducible texture occlusion to the image using a hash-based seed 
        (from the filename or image bytes).

        Args:
            img (PIL.Image or torch.Tensor): Input image.
            target: Optional target data (e.g., labels or bounding boxes).

        Returns:
            Tuple: Transformed image as torch.Tensor, and the unmodified target.
        """
                
        # Convert to PIL if needed
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Try to extract filename-based seed
        try:
            filename = img.filename  # Works if loaded via PIL.Image.open(path)
            seed = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % (2**32)
        except Exception:
            # If filename not available, hash first 1000 bytes of image content
            img_bytes = np.array(img).tobytes()[:1000]
            seed = int(hashlib.sha256(img_bytes).hexdigest(), 16) % (2**32)

        rng = random.Random(seed)

        if rng.random() > self.p or not self.obj_images:
            if isinstance(img, Image.Image):
                img = F.pil_to_tensor(img)
            return img, target

        # Choose plant
        obj_img = rng.choice(self.obj_images)

        # Resize randomly
        w, h = img.size
        scale = random.uniform(self.scale[0], self.scale[1])  # Random scale factor
        new_w, new_h = int(w * scale), int(w * scale)
        new_w, new_h = min(new_w, w), min(new_h, h)
        obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random rotation
        angle = rng.uniform(0, 360)
        obj_img = obj_img.rotate(angle, expand=True)
        # Make sure it fits into base image (w, h)
        new_w, new_h = obj_img.size
        if new_w >= w or new_h >= h:
            scale = min(w / new_w, h / new_h) * 0.8  # a bit smaller than max
            new_w, new_h = int(new_w * scale), int(new_h * scale)
            obj_img = obj_img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Random placement
        x_offset = rng.randint(0, w - new_w)
        y_offset = rng.randint(0, h - new_h)

        # Convert to NumPy
        obj_np = np.array(obj_img).astype(float)
        img_np = np.array(img).astype(float)

        # Alpha channel
        obj_alpha = obj_np[:, :, 3] > 128

        # Apply transparency
        # Blend the RGB channels with the alpha channel
        for c in range(3):  # Iterate over RGB channels
            img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = (
                obj_alpha * obj_np[:, :, c] + (1 - obj_alpha) * img_np[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
            )

        #img_occluded = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        # Convert back to PIL
        img_occluded = Image.fromarray(img_np.astype(np.uint8))

        return F.pil_to_tensor(img_occluded), target

