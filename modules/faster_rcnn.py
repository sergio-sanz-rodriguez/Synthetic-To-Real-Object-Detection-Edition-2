import torch
import torchvision
from typing import Union
from pathlib import Path
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

class StandardFasterRCNN(torch.nn.Module):

    """
        Creates a Faster Region-based CNN (RCNN) architecture using pytorch's predefined backbones for R-CNN: 'resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320'.
        More information is found in this link: https://pytorch.org/vision/master/models/faster_rcnn.html
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50", #['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        weights: Union[str, Path] = "DEFAULT",
        nms: list = [20, 5, 50, 2],
        hidden_layer: int = 256,

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

        """
        Ceates a Faster Region-based CNN (RCNN) architecture using predefined backbones: 'resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320'.
        - num_classes: Number of output classes for detection, excluding background (int). Default is 1.
        - weights: The pretrained weights to load for the backbone (str). Default is "DEFAULT".
        - backbone: Backbone architecture to use. Default is 'resnet50'. List of supported networks order by accuracy-speed tradefoof:
                    1. 'resnet50_v2: very high accuracy, moderate-high speed
                    2. 'resnet50': high accuracy, moderate speed
                    3. 'mobilenet_v3_large': moderate accuracy, very high speed
                    4. 'mobilenet_v3_large_320': moderate-high accuracy, very high speed
                    ['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        - nms (Non-Maximum Suppression): A list with the following values:
            - pre_nms_top_n_train: Number of proposals before NMS algorithm the RPN will generate during training.
            - post_nms_top_n_train: Number of proposal to be reatined after applying NMS during training.
            - pre_nms_top_n_test: Same as pre_nms_top_n_train but for testing.
            - post_nms_top_n_test: Same as post_nms_top_n_train but for testing.
            Default: None with default numbers. E.g. [20, 5, 50, 2]
        - hidden_layer: Number of hidden units for the mask prediction head. Default is 256.
        - device: Target device: GPU or CPU
        """

        super().__init__()
        
        # Check if the specified backbone is available
        backbone_list = ['resnet50', 'resnet50_v2', 'mobilenet_v3_large', 'mobilenet_v3_large_320']
        assert backbone in backbone_list, f"[ERROR] Backbone '{backbone}' not recognized."

        assert isinstance(num_classes, int), "[ERROR] num_classes must be an integer."
        assert isinstance(hidden_layer, int) and hidden_layer > 0, "[ERROR] hidden_layer must be a positive integer."

        # Load default pretrained weights if "DEFAULT" or None
        if backbone == 'resnet50':
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        elif backbone == 'resnet50_v2':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        elif backbone == 'mobilenet_v3_large':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        else:
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        
        # Replace the classification head (bounding box predictor)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor (for Mask R-CNN models)
        if "maskrcnn" in self.model.__class__.__name__.lower():
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
            
        # Move the model to the specified device
        self.model.to(device)

        # Load custom weights if provided
        if isinstance(weights, (str, Path)) and weights != "DEFAULT":
            weights_path = Path(weights)
            if weights_path.exists() and weights_path.suffix == '.pth':
                # Load the custom weights
                checkpoint = torch.load(weights_path, map_location=device)
                # Update the model with the checkpoint's state_dict
                self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"[ERROR] Custom weights path '{weights}' is not valid or does not point to a valid checkpoint file.")
        
        # === TUNED FOR SINGLE-OBJECT DETECTION ===

        # RPN proposals
        self.model.rpn.pre_nms_top_n_train = nms[0]
        self.model.rpn.post_nms_top_n_train = nms[1]
        self.model.rpn.pre_nms_top_n_test = nms[2]
        self.model.rpn.post_nms_top_n_test = nms[3]

        # ROI head settings        
        #self.model.roi_heads.detections_per_img = 1  # Only keep top-1 box
        #self.model.roi_heads.nms_thresh = 0.3        # Aggressive NMS to reduce similar boxes

        # === OPTIONAL: Custom anchor generator (if you know object scale/shape) ===
        #from torchvision.models.detection.rpn import AnchorGenerator
        #custom_anchor_gen = AnchorGenerator(
        #    sizes=((64,),),  # Custom size
        #     aspect_ratios=((1.0,),)
        # )
        #self.model.rpn.anchor_generator = custom_anchor_gen
        
    def forward(self, images, targets=None):

        """
        Forward pass through the model:
        - images: Input images (tensor or list of tensors).
        - targets: Ground truth targets for training (optional, only needed for training).
        """
        
        return self.model(images, targets)


