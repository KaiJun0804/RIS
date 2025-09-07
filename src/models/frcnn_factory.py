import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_faster_rcnn_model(num_classes: int = 2, use_pretrained: bool = True):
    """Create Faster R-CNN ResNet50 FPN for a given number of classes.

    Args:
        num_classes: Number of classes including background (for plates: 2 -> background + plate)
        use_pretrained: If True, load COCO pretrained weights then replace the predictor
    """
    if use_pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model