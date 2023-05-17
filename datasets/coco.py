import os
from io import BytesIO

import requests
import torch.utils.data
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms as T

from models.SSD.ssd.data.datasets import build_dataset
from models.SSD.ssd.data.transforms import build_transforms, build_target_transform
from models.SSD.ssd.structures.container import Container
from models.SSD.ssd.config import cfg

class_names = ('__background__',
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush')


def get_coco_dataset(is_train, preprocessing=True):
    config_file = "models/SSD/configs/vgg_ssd512_coco_trainval35k.yaml"
    print(f"Loading DATASET with config file {config_file}.")
    cfg.merge_from_file(config_file)
    cfg.freeze()
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(COCODataset, dataset_list, path="datasets/coco/data",
                             transform=train_transform, target_transform=target_transform,
                             is_train=is_train, preprocessing=preprocessing)
    return datasets[0]


def coco_train(preprocessing=True):
    return get_coco_dataset(is_train=True, preprocessing=preprocessing)


def coco_test(preprocessing=True):
    return get_coco_dataset(is_train=False, preprocessing=preprocessing)


class COCODataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, data_dir, ann_file, transform=None, target_transform=None, remove_empty=False, preprocessing=True):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        if self.remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

        self.reverse_normalization = T.Normalize(mean=[-123, -117, -104], std=[1, 1, 1])

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        # print(torch.tensor(targets['labels']))
        targets = torch.tensor(targets['labels'])
        return image, targets[:, None].expand(targets.shape[0], 2)

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data

    def _read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_input_image(self, sample_id: int) -> Image:
        """
        Returns input image.
        Args:
            loader: dataloader
            sample_id: sample id

        Returns: Image

        """
        img_info = self.get_img_info(sample_id)
        response = requests.get(img_info['coco_url'])
        return Image.open(BytesIO(response.content))

    def get_bbox_pred(self, bbox_output: Tensor, bbox_id: int, image_dim: tuple) -> Tensor:
        """
        Retrieves bounding box and rescales to shape of input image
        Args:
            bbox_output: model output containing bounding boxes (batch_size, #number of bb, 4)
            bbox_id: id of bbox
            image_dim: dimension of input image (w, h)

        Returns: bounding box coordinates (xmin, ymin, xmax, ymax)

        """

        bb_box = bbox_output[:, bbox_id, :][0]
        bb_box[0::2] *= image_dim[0]
        bb_box[1::2] *= image_dim[1]
        return bb_box

    def get_bbox_gt(self, sample_id: int, bb_class: int) -> Tensor:
        """
        Returns ground truth bounding box of class bb_class for sample sample_id.
        Args:
            loader: dataloader
            sample_id: sample id
            bb_class: class of bb

        Returns: bounding box coordinates (xmin, ymin, xmax, ymax)

        """
        _, annotation = self.get_annotation(sample_id)
        return annotation[0][list(annotation[1]).index(bb_class)]
