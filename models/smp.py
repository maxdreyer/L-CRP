import segmentation_models_pytorch as smp
import torch

SMP_MODELS = {
    "unet": smp.Unet,
}

CONFIGS = {
    "unet": {
        "in_channels": 3,
        "encoder_name": "vgg13",
        "encoder_weights": "imagenet",
        "ckpt_path": "models/smp_unet_vgg13_cityscapes.pth"
    }
}


def get_smp(model_name: str):
    def get_model(**kwargs):
        cfg = CONFIGS[model_name]
        model_kwargs = {
            k: kwargs[k] if k in kwargs.keys() else cfg[k]
            for k in ["in_channels", "classes", "encoder_name", "encoder_weights"]
        }
        model = SMP_MODELS[model_name](**model_kwargs)

        if "ckpt_path" in kwargs:
            print("Loaded checkpoint", kwargs["ckpt_path"])
            model.load_state_dict(torch.load(kwargs["ckpt_path"]))
        elif "ckpt_path" in cfg:
            print("Loaded checkpoint", cfg["ckpt_path"])
            model.load_state_dict(torch.load(cfg["ckpt_path"]))

        return model

    return get_model