import os
from typing import Dict, List, Any

import PIL
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
from crp.helper import max_norm
from crp.image import imgify, get_crop_range
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import gaussian_blur
from zennit.core import stabilize
from zennit.image import gridify
from zennit.image import imsave as imsave_zennit


def save_img(data: torch.Tensor, name: str, dataset_name: str, other_dir: str = None, norm: bool = False):
    """
    Plot and save explanation heat map.
    Args:
        explanation: gradient/attribution tensor
        name: name of file
        dataset_name: dataset name

    """
    data = np.array(data.detach().cpu())

    # normalization of data
    if norm:
        max_val = np.abs(data).max((1, 2), keepdims=True)
        data = data / 2 / (max_val + 1e-12) + 0.5
    grid = gridify(data, fill_value=0.5)

    if other_dir is None:
        dir = "images"
    else:
        dir = other_dir

    os.makedirs(f"results/{dir}/{dataset_name}", exist_ok=True)
    imsave_zennit(f"results/{dir}/{dataset_name}/{name}.png",
                  grid,
                  vmin=0.,
                  vmax=1.,
                  level=1.0,
                  cmap='bwr')


def plot_grid(ref_c: Dict[int, List[torch.Tensor]], cmap="bwr", vmin=None, vmax=None, symmetric=False,
              resize=(224, 224), padding=True, figsize=None, dpi=100) -> None:
    nrows = len(ref_c)
    ncols = len(next(iter(ref_c.values())))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=0.05)

    for r, key in enumerate(ref_c):

        for c, img in enumerate(ref_c[key]):

            img = imgify(img, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding)

            if nrows == 1:
                ax = axs[c]
            elif ncols == 1:
                ax = axs[r]
            else:
                ax = axs[r, c]

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(key)


def gauss_p_norm(x: Any, sigma: int = 6) -> Any:
    """ Applies Gaussian filter and normalizes"""
    return normalize(gaussian_filter(x, sigma=sigma))


def normalize(a: Any) -> Any:
    """ Applies normalization"""
    return a / a.max()


def mask_img(img: torch.Tensor, mask: torch.Tensor, alpha: int = 0.5) -> torch.Tensor:
    """ Masks input sample with mask"""
    minv = 1 - mask  # inverse mask
    return img * mask + img * minv * alpha


def get_masked(imgs: Dict, hms: Dict, thresh=0.2):
    """ Masks img dict from CRP library using heatmaps dict."""
    return {k: [mask_img(img.to(hm), gauss_p_norm(hm) > thresh) for img, hm in zip(imgs[k], hms[k])] for k in
            imgs.keys()}


def get_masks(hms: Dict, thresh=0.2):
    """ Masks img dict from CRP library using heatmaps dict."""
    return {k: [gauss_p_norm(hm) > thresh for hm in zip(hms[k])] for k in
            hms.keys()}


def slice_img(img: Any, pad_x: int, pad_y: int) -> Any:
    if pad_x == 0 and pad_y == 0:
        return img
    if len(img.shape) == 3:
        return img[:, pad_y:-pad_y] if pad_y else img[..., pad_x:-pad_x]
    elif len(img.shape) == 2:
        return img[pad_y:-pad_y] if pad_y else img[..., pad_x:-pad_x]


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, dpi=200)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def stroke(origin_image, threshold, stroke_size, colors):
    img = np.array(origin_image)
    h, w, _ = img.shape
    padding = stroke_size + 50
    alpha = img[:, :, 0] * 0 + 1.0
    rgb_img = img[:, :, 0:3]
    bigger_img = cv2.copyMakeBorder(rgb_img, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    alpha = cv2.copyMakeBorder(alpha, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    bigger_img = cv2.merge((bigger_img, alpha))
    h, w, _ = bigger_img.shape

    _, alpha_without_shadow = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)  # threshold=0 in photoshop
    alpha_without_shadow = 255 - alpha_without_shadow
    dist = cv2.distanceTransform(alpha_without_shadow, cv2.DIST_L2, cv2.DIST_MASK_3)  # dist l1 : L1 , dist l2 : l2
    stroked = change_matrix(dist, stroke_size)
    stroke_alpha = (stroked * 255).astype(np.uint8)

    stroke_b = np.full((h, w), colors[0][2], np.uint8)
    stroke_g = np.full((h, w), colors[0][1], np.uint8)
    stroke_r = np.full((h, w), colors[0][0], np.uint8)

    stroke = cv2.merge((stroke_b, stroke_g, stroke_r, stroke_alpha))
    stroke = cv2pil(stroke)
    bigger_img = cv2pil(bigger_img)
    result = Image.alpha_composite(stroke, bigger_img)
    return result


def change_matrix(input_mat, stroke_size):
    stroke_size = stroke_size - 1
    mat = np.ones(input_mat.shape)
    check_size = stroke_size + 1.0
    mat[input_mat > check_size] = 0
    border = (input_mat > stroke_size) & (input_mat <= check_size)
    mat[border] = 1.0 - (input_mat[border] - stroke_size)
    return mat


def cv2pil(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(cv_img.astype("uint8"))
    return pil_img


@torch.no_grad()
def vis_opaque_img_border(data_batch, heatmaps, rf=False, alpha=0.5, vis_th=0.05, crop_th=0.01,
                          kernel_size=13) -> Image.Image:
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th.
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.

    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
        filtered_heat = filtered_heat / filtered_heat.clamp(min=0).max()
        vis_mask = filtered_heat > vis_th

        if rf:
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)

            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                # check whether img_t or vis_mask_t is not empty
                img = img_t
                vis_mask = vis_mask_t

        inv_mask = ~vis_mask
        outside = (img * vis_mask).sum((1, 2)).mean(0) / stabilize(vis_mask.sum()) > 0.5

        img = img * vis_mask + img * inv_mask * alpha + outside * 0 * inv_mask * (1 - alpha)

        img = imgify(img.detach().cpu()).convert('RGBA')

        img_ = np.array(img).copy()
        img_[..., 3] = (vis_mask * 255).detach().cpu().numpy().astype(np.uint8)
        img_ = mystroke(Image.fromarray(img_), 1, color='white' if outside else 'white')

        img.paste(img_, (0, 0), img_)

        imgs.append(img.convert('RGB'))

    return imgs


def mystroke(img, size: int, color: str = 'black'):

    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    stroke = Image.new(img.mode, img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke)
    fill = (0, 0, 0, 180) if color == 'black' else (255, 255, 255, 180)
    for x in range(X):
        for y in range(Y):
            if edge[x, y][3] > 0:
                draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)
    stroke.paste(img, (0, 0), img)

    return stroke
