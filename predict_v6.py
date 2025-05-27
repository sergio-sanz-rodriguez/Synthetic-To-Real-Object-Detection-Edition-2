#from ultralytics import YOLO
from pathlib import Path
import os
import yaml
import torch
import argparse
import torchvision.ops as ops
from pathlib import Path
from torchvision.io import decode_image
from torchvision.transforms import v2 as T
from modules.faster_rcnn import StandardFasterRCNN

def load_model(model: torch.nn.Module, target_dir: str, model_name: str, device: torch.device):
        
    """Loads a PyTorch model from a target directory.

    Args:
        model: A target PyTorch model to load.
        target_dir: A directory where the model is located.
        model_name: The name of the model to load. Should include
        ".pth", ".pt", ".pkl", ".h5", or ".torch" as the file extension.

    Returns:
        The loaded PyTorch model.
    """

    # Define the list of valid extensions
    valid_extensions = [".pth", ".pt", ".pkl", ".h5", ".torch"]

    # Create model save path
    assert any(model_name.endswith(ext) for ext in valid_extensions), f"model_name should end with one of {valid_extensions}"
    model_save_path = Path(target_dir) / model_name

    # Load the model
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    return model

# Pre-processing transformations
def get_transform():

    """
    Returns a composition of transformations for preprocessing images.
    """

    transforms = []   
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# Function to remove redundant boxes and masks
def prune_predictions(pred, image_area):

    """
    Filters out redundant predictions.

    Args:
        pred: The raw predictions containing "boxes", "scores", "labels", and "masks".        

    Returns:
        A dictionary with filtered and refined predictions:
            "boxes": Tensor of kept bounding boxes.
            "scores": Tensor of kept scores.
            "labels": Tensor of kept labels.
    """

    # Remove large unusual bounding boxes
    max_box_area = image_area * 0.5
    areas = (pred["boxes"][:, 2] - pred["boxes"][:, 0]) * (pred["boxes"][:, 3] - pred["boxes"][:, 1])
    keep_idx = areas < max_box_area
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
    high_conf_idx = scores > 0.8

    # Extract the best bounding box, score, and label
    best_pred = {
        "boxes": pred["boxes"][best_idx].unsqueeze(0).long(), 
        "scores": pred["scores"][best_idx].unsqueeze(0),
        "labels": pred["labels"][best_idx].unsqueeze(0),
    }

    filtered_pred = {
        "boxes":  pred["boxes"][high_conf_idx].long(),
        "scores": pred["scores"][high_conf_idx],
        "labels": pred["labels"][high_conf_idx],
    }

    # Apply Non-Maximum Suppression (NMS) to remove overlapping predictions
    if len(filtered_pred["boxes"]) == 0:
        if len(best_pred["boxes"]) > 0:
            return best_pred
        else:
            return filtered_pred 
    
    keep_idx = ops.nms(filtered_pred["boxes"].float(), filtered_pred["scores"], 0.01)

    # Return filtered predictions
    keep_preds = {
        "boxes": filtered_pred["boxes"][keep_idx],
        "scores": filtered_pred["scores"][keep_idx],
        "labels": filtered_pred["labels"][keep_idx],
    }

    # Ensure the best prediction is always included
    best_box = best_pred["boxes"][0]
    if not any(torch.equal(best_box, box) for box in keep_preds["boxes"]):
        keep_preds["boxes"] = torch.cat([keep_preds["boxes"], best_pred["boxes"]])
        keep_preds["scores"] = torch.cat([keep_preds["scores"], best_pred["scores"]])
        keep_preds["labels"] = torch.cat([keep_preds["labels"], best_pred["labels"]])

    # If we have a set of good candidates, let's select the bounding box with the highest score
    if keep_preds["boxes"].shape[0] > 1:

        # Return only the one with the highest score            
        idx = keep_preds['scores'].argmax().item()
        final_pred = {
            "boxes": keep_preds["boxes"][idx].unsqueeze(0),
            "scores": keep_preds["scores"][idx].unsqueeze(0),
            "labels": keep_preds["labels"][idx].unsqueeze(0),
        }
        return final_pred
    
    return keep_preds


def ensemble_predictions(predictions):

    """
    Ensemble method that takes multiple predictions and selects the best one based on the maximum score.

    Args:
        predictions (list of dict): each dict has "boxes", "scores", "labels".

    Returns:
        dict: single prediction with "boxes", "scores", "labels"
    """

    # Filter out empty predictions and keep track of their index
    preds = [(i+1, p) for i, p in enumerate(predictions) if p is not None and p["boxes"].nelement() > 0]

    if not preds:
        return {}

    # Compute max score per prediction
    max_scores = [(p["scores"].max(), i, p) for i, p in preds]

    # Print all scores
    #print(" ".join(f"pred{i}: {score.item():.4f}" for score, i, _ in max_scores))

    # Select best prediction
    best_score, best_i, best_pred = max(max_scores, key=lambda x: x[0])
    #print(f"pred{best_i}: {best_score.item():.4f}")

    return best_pred, best_i, best_score

def merge_and_select(predictions):
    
    """
    Merges predictions from multiple models, groups overlapping bounding boxes by IoU,
    and selects the highest-scoring bounding box from the most frequently detected object.

    Args:
        predictions (list of dict): List of prediction dictionaries, each containing
            'boxes' (Tensor[N, 4]), 'scores' (Tensor[N]), and 'labels' (Tensor[N]).

    Returns:
        dict: A dictionary with a single selected prediction:
            - 'boxes': Tensor[1, 4]
            - 'scores': Tensor[1]
            - 'labels': Tensor[1]
    """
    
    # Initialize containers to merge predictions
    all_boxes = []
    all_scores = []
    all_labels = []

    # Append boxes, scores, and labels from each prediction dictionary
    for pred in predictions:
        all_boxes.append(pred["boxes"])
        all_scores.append(pred["scores"])
        all_labels.append(pred["labels"])

    # Concatenate all predictions into single tensors
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Return empty prediction if no boxes are found
    num_boxes = all_boxes.size(0)
    if num_boxes == 0:
        return {
            "boxes": torch.empty((1, 4)),
            "scores": torch.tensor([0.0]),
            "labels": torch.tensor([0])
        }

    # Compute IoU matrix between all pairs of boxes
    iou_matrix = ops.box_iou(all_boxes, all_boxes)

    # Cluster boxes with IoU greater than the threshold
    clusters = []
    visited = set()
    for i in range(num_boxes):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, num_boxes):
            if j not in visited and iou_matrix[i, j] > 0.01:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    # Find the largest cluster(s) (most frequently detected object)
    max_len = max(len(c) for c in clusters)
    candidate_clusters = [c for c in clusters if len(c) == max_len]

    # Among the candidates, select the one with the highest score
    best_cluster = max(
        candidate_clusters,
        key=lambda c: all_scores[c].max().item()
    )
    
    # From the best cluster, select the bounding box with the highest score
    cluster_scores = all_scores[best_cluster]
    best_idx_in_cluster = best_cluster[torch.argmax(cluster_scores).item()]

    # Extract the final best prediction
    best_box = all_boxes[best_idx_in_cluster].unsqueeze(0)
    best_score = all_scores[best_idx_in_cluster].unsqueeze(0)
    best_label = all_labels[best_idx_in_cluster].unsqueeze(0)

    # Return the final prediction
    final_pred = {
        "boxes": best_box,
        "scores": best_score,
        "labels": best_label
    }

    return final_pred

def same_object(pred1, pred2):

    """
    Compare two single-box predictions and return whether they refer to the same object.

    Args:
        pred1, pred2 (dict): Each with "boxes" (Tensor [1, 4]), "scores", "labels".
        iou_threshold (float): IoU threshold to consider them the same.

    Returns:
        bool: True if IoU > threshold, else False.
    """

    iou = ops.box_iou(pred1["boxes"], pred2["boxes"])[0, 0]

    return iou > 0.01 and pred1["labels"][0] == pred2["labels"][0]


def box_area(pred):

    """
    Compute the area of a bounding box

    Args:
        pred: bounding box with [x1, y1, x2, y2]
    
    Returns:
        area
    """
    
    box = pred["boxes"][0]
    return (box[2] - box[0]) * (box[3] - box[1])

# Function to predict and save images
def predict_and_save(model_list, image_path, output_path_txt, device):

    """
    Predict bounding boxes using a custom model and save them in YOLO format.
    
    Args:
        model_list: list of object detection models.
        image_path: path of the image
        img_width: Width of the image.
        img_height: Height of the image.
        output_path_txt: Path to save the predictions.
    """

    # Load and transform image
    image = decode_image(image_path)
    img_height, img_width = image.shape[1], image.shape[2]
    img_area = img_height * img_width
    transform = get_transform()
    x = transform(image)[:3, ...].to(device)

    # Set all models to eval mode and move to device
    for model in model_list:
        model.eval().to(device)

    # Predict with each model
    with torch.no_grad():
        predictions = []
        for model in model_list:
            pred = model([x, ])[0]
            if pred["boxes"].nelement() > 0:
                pred = prune_predictions(pred, img_area)
            predictions.append(pred)

    # Ensemble the predictions
    predA = merge_and_select(predictions[:-1])
    predB = predictions[-1]
    if same_object(predA, predB) and box_area(predB) <= box_area(predA):
        final_pred = predB
        print(f"pred4: {final_pred['scores'].item():.4f}")
    else:
        final_pred = predA
        names = ['pred1', 'pred2', 'pred3']
        for p, name in zip(predictions[:-1], names):                
            if torch.equal(p['boxes'], predA['boxes']):
                best_pred= name
        print(f"{best_pred}: {final_pred['scores'].item():.4f}")

    # Save the final prediction in YOLO format
    boxes = final_pred["boxes"]
    scores = final_pred["scores"]
    labels = final_pred["labels"]

    with open(output_path_txt, 'w') as f:

        for box, score, label in zip(boxes, scores, labels):
            cls_id = int(label.item()) - 1 # Always zero
            conf = float(score.item())
            xmin, ymin, xmax, ymax = box.tolist()
            
            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            f.write(f"{cls_id} {conf} {x_center} {y_center} {width} {height}\n")

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Object Detection Kaggle Competition")    
    parser.add_argument("--modelA", type=str, action='append', required=True, help="Paths to modelA files (use --model path multiple times)")
    parser.add_argument("--modelB", type=str, required=True, help="Paths to modelB file")

    args = parser.parse_args()

    # Set working directory
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load test path from YAML
    with open(this_dir / 'yolo_params_2.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' not in data or not data['test']:
            print("Add 'test: path/to/test/images' to yolo_params_2.yaml")
            exit()
        images_dir = Path(data['test'])
    
    # Validate test directory
    if not images_dir.exists():
        print(f"Test directory {images_dir} does not exist")
        exit()
    if not any(images_dir.glob('*')):
        print(f"Test directory {images_dir} is empty")
        exit()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate and load models
    models = []
    for modelA_path in args.modelA:
        model = StandardFasterRCNN(backbone="resnet50_v2", num_classes=2, device=device, nms = [20, 5, 50, 2])
        model = load_model(model, Path(modelA_path).parent, Path(modelA_path).name, device)
        models.append(model)
    
    modelB_path = args.modelB
    model = StandardFasterRCNN(backbone="resnet50_v2", num_classes=2, device=device, nms = [20, 5, 50, 2])
    model = load_model(model, Path(modelB_path).parent, Path(modelB_path).name, device)
    models.append(model)
    
    # Directory with images to generate predictions
    output_dir = this_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create labels subdirectories
    labels_output_dir = output_dir / 'labels'
    
    # images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg','.jpeg']:
            continue
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name  # Save label in 'labels' folder
        predict_and_save(models, img_path, output_path_txt, device)

    print(f"Bounding box labels saved in {labels_output_dir}")

if __name__ == '__main__':
    main()