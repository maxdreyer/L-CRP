from models.SSD.ssd.modeling import registry
from models.SSD.ssd.modeling.backbone.vgg import VGG
from models.SSD.ssd.modeling.backbone.mobilenet import MobileNetV2
from models.SSD.ssd.modeling.backbone.efficient_net import EfficientNet
from models.SSD.ssd.modeling.backbone.mobilenetv3 import MobileNetV3

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'MobileNetV3']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
