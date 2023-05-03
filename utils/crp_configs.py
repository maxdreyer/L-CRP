from zennit.torchvision import ResNetCanonizer

from utils.crp import CondAttributionLocalization, CondAttributionSegmentation, FeatureVisualizationLocalization, \
    FeatureVisualizationSegmentation
from utils.zennit_canonizers import YoloV5V6Canonizer, DeepLabV3PlusCanonizer
from utils.zennit_composites import EpsilonPlusFlat, EpsilonGammaFlat

COMPOSITES = {
    # object detectors
    "yolov5": EpsilonPlusFlat,
    "yolov6": EpsilonGammaFlat,
    # segmentation models
    "unet": EpsilonPlusFlat,
    "deeplabv3plus": EpsilonPlusFlat,
}

CANONIZERS = {
    # object detectors
    "yolov5": YoloV5V6Canonizer,
    "yolov6": YoloV5V6Canonizer,
    # segmentation models
    "unet": ResNetCanonizer,
    "deeplabv3plus": DeepLabV3PlusCanonizer,
}

ATTRIBUTORS = {
    # object detectors
    "yolov5": CondAttributionLocalization,
    "yolov6": CondAttributionLocalization,
    # segmentation models
    "unet": CondAttributionSegmentation,
    "deeplabv3plus": CondAttributionSegmentation,
}

VISUALIZATIONS = {
    # object detectors
    "yolov5": FeatureVisualizationLocalization,
    "yolov6": FeatureVisualizationLocalization,
    # segmentation models
    "unet": FeatureVisualizationSegmentation,
    "deeplabv3plus": FeatureVisualizationSegmentation,
}