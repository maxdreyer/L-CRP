import click
import torch
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

from datasets import get_dataset
from models import get_model
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES


@click.command()
@click.option("--model_name", default="yolov6")
@click.option("--dataset_name", default="coco2017")
# @click.option("--model_name", default="unet")
# @click.option("--dataset_name", default="cityscapes")
@click.option("--batch_size", default=12)
def main(model_name, dataset_name, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset(preprocessing=True)
    model = get_model(model_name=model_name, classes=n_classes)

    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])

    model = model.to(device)
    model.eval()
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    layer_map = {layer: cc for layer in layer_names}

    attribution = ATTRIBUTORS[model_name](model)

    fv = VISUALIZATIONS[model_name](attribution,
                                    dataset,
                                    layer_map,
                                    preprocess_fn=lambda x: x,
                                    path=f"output/crp/{model_name}_{dataset_name}",
                                    max_target="max")

    fv.run(composite, 0, len(dataset), batch_size, 100)


if __name__ == "__main__":
    main()
