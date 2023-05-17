import torch

from models.deeplabv3plus import get_deeplabv3plus
from models.smp import get_smp
from models.yolov5 import get_yolov5
from models.yolov6 import get_yolov6
from models.ssd import get_ssd
MODELS = {
    # object detectors
    "yolov5": get_yolov5,
    "yolov6": get_yolov6,
    "ssd": get_ssd,
    # segmentation models
    "unet": get_smp("unet"),
    "deeplabv3plus": get_deeplabv3plus,
}

def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name in MODELS:
        model = MODELS[model_name](**kwargs)
        return model
    else:
        print(f"Model {model_name} not available")
        exit()
