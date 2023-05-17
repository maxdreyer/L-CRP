from typing import Tuple, Any

import torch
from torch import Tensor
from torch.nn import Module

from models.SSD.ssd.modeling.detector import build_detection_model
from models.SSD.ssd.config import cfg
from models.SSD.ssd.utils.checkpoint import CheckPointer


def get_ssd(**kwargs):
    config_file = "models/SSD/configs/vgg_ssd512_coco_trainval35k.yaml"
    print(f"Loading SSD from config file {config_file}.")
    cfg.merge_from_file(config_file)
    cfg.freeze()
    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=None)
    checkpointer.load("models/checkpoints/vgg_ssd512_coco_trainval35k.pth", use_latest=False)
    return SSDWrapper(model)


class SSDWrapper(torch.nn.Module):

    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def forward(self, data: Tensor) -> Tensor:
        out = self.model.forward(data)
        return torch.cat([out[1], out[0]], dim=2)

    def predict_with_boxes(self, data: Tensor) -> Tuple[Any, Any]:
        out = self.model.forward(data)
        boxes = out[0] * 512
        return out[1], boxes
