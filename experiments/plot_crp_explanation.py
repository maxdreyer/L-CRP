import copy

import click
import numpy as np
import torch
from crp.helper import get_layer_names
from crp.image import imgify
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms.functional as F

from datasets import get_dataset
from models import get_model
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes, make_grid
import zennit.image as zimage

from utils.crp import ChannelConcept
from utils.crp_configs import ATTRIBUTORS, CANONIZERS, VISUALIZATIONS, COMPOSITES
from utils.render import vis_opaque_img_border

EXAMPLES = [{
        "model_name": "deeplabv3plus",
        "dataset_name": "voc2012",
        "sample_id": 401,
        "class_id": 13,
        "layer": "backbone.layer4.0.conv3"},
    {
        "model_name": "unet",
        "dataset_name": "cityscapes",
        "sample_id": 22,
        "class_id": 12,
        "layer": "encoder.features.15"},
    {
        "model_name": "yolov6",
        "dataset_name": "coco2017",
        "sample_id": 140,
        "class_id": 0,
        "layer": "backbone.ERBlock_5.0.rbr_dense.conv"},
    {
        "model_name": "yolov5",
        "dataset_name": "coco2017",
        "sample_id": 140,
        "class_id": 0,
        "layer": "model.8.cv3.conv"},
    {
        "model_name": "ssd",
        "dataset_name": "coco",
        "sample_id": 195,
        "class_id": 1,
        "layer": "model.backbone.vgg.28"},
]

EXAMPLE = -1

@click.command()
@click.option("--model_name", default=EXAMPLES[EXAMPLE]["model_name"])
@click.option("--dataset_name", default=EXAMPLES[EXAMPLE]["dataset_name"])
@click.option("--sample_id", default=EXAMPLES[EXAMPLE]["sample_id"], type=int)
@click.option("--class_id", default=EXAMPLES[EXAMPLE]["class_id"])
@click.option("--layer", default=EXAMPLES[EXAMPLE]["layer"], type=str)
@click.option("--prediction_num", default=0)
@click.option("--mode", default="relevance")
@click.option("--n_concepts", default=3)
@click.option("--n_refimgs", default=12)
def main(model_name, dataset_name, sample_id, class_id, layer, prediction_num, mode, n_concepts, n_refimgs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_dataset, n_classes = get_dataset(dataset_name=dataset_name).values()
    dataset = test_dataset(preprocessing=True)
    model = get_model(model_name=model_name, classes=n_classes)

    model = model.to(device)
    model.eval()

    attribution = ATTRIBUTORS[model_name](model)
    composite = COMPOSITES[model_name](canonizers=[CANONIZERS[model_name]()])
    condition = [{"y": class_id}]

    img, t = dataset[sample_id]
    img = img[None, ...].to(device)
    ratio = img.shape[-2] / img.shape[-1]

    fig, axs = plt.subplots(n_concepts, 3,
                            figsize=((1.6 + 1 / ratio) * n_refimgs / 4, 1.6 * n_concepts),
                            gridspec_kw={'width_ratios': [4, 4, n_refimgs * ratio]},
                            dpi=200)

    if "deeplab" in model_name or "unet" in model_name:

        attr = attribution(copy.deepcopy(img).requires_grad_(), condition, composite, record_layer=[layer],
                           init_rel=1)
        heatmap = zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True)

        mask = (attr.prediction[0].argmax(dim=0) == class_id).detach().cpu()
        sample_ = dataset.reverse_augmentation(img[0] + 0)
        img_ = F.to_pil_image(draw_segmentation_masks(sample_, masks=mask, alpha=0.5, colors=["red"]))

        axs[0][0].imshow(np.asarray(img_))
        axs[0][0].contour(mask, colors="black", linewidths=[2])


    elif "yolo" in model_name or "ssd" in model_name:
        attribution.take_prediction = prediction_num
        attr = attribution(img.requires_grad_(), condition, composite, record_layer=[layer], init_rel=1)
        heatmap = np.array(zimage.imgify(attr.heatmap.detach().cpu(), symmetric=True))
        heatmap = zimage.imgify(heatmap, symmetric=True)

        predicted_boxes = model.predict_with_boxes(img)[1][0]
        predicted_classes = attr.prediction.argmax(dim=2)[0]
        print("Predicted classes: ", torch.unique(predicted_classes).detach().cpu().numpy())
        sorted = attr.prediction.max(dim=2)[0].argsort(descending=True)[0]
        predicted_classes = predicted_classes[sorted]
        predicted_boxes = predicted_boxes[sorted]
        predicted_boxes = [b for b, c in zip(predicted_boxes, predicted_classes) if c == class_id][prediction_num]
        boxes = torch.tensor(predicted_boxes, dtype=torch.float)[None]
        colors = ["#ffcc00" for _ in boxes]
        result = draw_bounding_boxes((dataset.reverse_normalization(img[0])).type(torch.uint8),
                                     boxes, colors=colors, width=8)

        img_ = F.to_pil_image(result)
        attribution.take_prediction = 0

        axs[0][0].imshow(np.asarray(img_))
    else:
        raise NameError

    layer_map = get_layer_names(model, [torch.nn.Conv2d])
    fv = VISUALIZATIONS[model_name](attribution, dataset, layer_map, preprocess_fn=lambda x: x,
                                    path=f"output/crp/{model_name}_{dataset_name}",
                                    max_target="max", device=device)

    if mode == "relevance":
        channel_rels = ChannelConcept().attribute(attr.relevances[layer], abs_norm=True)
    else:
        channel_rels = attr.activations[layer].detach().cpu().flatten(start_dim=2).max(2)[0]
        channel_rels = channel_rels / channel_rels.abs().sum(1)[:, None]

    topk = torch.topk(channel_rels[0], n_concepts)
    topk_ind = topk.indices.detach().cpu().numpy()
    topk_rel = topk.values.detach().cpu().numpy()

    print("Concepts:", topk)

    conditions = [{"y": class_id, layer: c} for c in topk_ind]
    if mode == "relevance":
        attribution.take_prediction = prediction_num
        cond_heatmap, _, _, _ = attribution(img.requires_grad_(), conditions, composite, exclude_parallel=True)
        attribution.take_prediction = 0
    else:
        cond_heatmap = torch.stack([attr.activations[layer][0][t] for t in topk_ind]).detach().cpu()

    print("Computing reference images...")
    ref_imgs = fv.get_max_reference(topk_ind, layer, mode, (0, n_refimgs), composite=composite, rf=True,
                                    plot_fn=vis_opaque_img_border, batch_size=4)

    print("Plotting...")
    resize = torchvision.transforms.Resize((150, 150))

    for r, row_axs in enumerate(axs):

        for c, ax in enumerate(row_axs):
            if c == 0:
                if r == 0:
                    ax.set_title(f"input")
                elif r == 1:
                    ax.set_title("heatmap")
                    ax.imshow(heatmap)
                else:
                    ax.axis('off')

            if c == 1:
                if r == 0:
                    ax.set_title("cond. heatmap")
                ax.imshow(imgify(cond_heatmap[r], symmetric=True, cmap="bwr", padding=False))
                ax.set_ylabel(f"concept {topk_ind[r]}\n relevance: {(topk_rel[r] * 100):2.1f}%")

            elif c >= 2:
                if r == 0 and c == 2:
                    ax.set_title("concept visualizations")
                grid = make_grid(
                    [resize(torch.from_numpy(np.array(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[r]]],
                    nrow=int(n_refimgs / 2),
                    padding=0)
                grid = np.array(zimage.imgify(grid.detach().cpu()))
                ax.imshow(grid)
                ax.yaxis.set_label_position("right")

            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    print()
    plt.show()
    print("Done plotting.")


if __name__ == "__main__":
    main()
