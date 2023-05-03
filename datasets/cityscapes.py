import sys
import torch.utils.data
import torch
import torch.utils.data

import numpy as np
import cv2
import os
import albumentations as A
import torchvision
from PIL import Image
from torchvision.transforms import transforms

train_transforms = A.Compose([
    A.Resize(height=256, width=512),
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(),
    A.GaussNoise(),
    A.Normalize(),
])

test_transforms = A.Compose([
    A.Resize(height=256, width=512),
    A.Normalize(),
])

train_dirs = ["jena/", "zurich/", "weimar/", "ulm/", "tubingen/", "stuttgart/",
              "strasbourg/", "monchengladbach/", "krefeld/", "hanover/",
              "hamburg/", "erfurt/", "dusseldorf/", "darmstadt/", "cologne/",
              "bremen/", "bochum/", "aachen/"]
val_dirs = ["frankfurt/", "munster/", "lindau/"]
test_dirs = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]

def cityscapes_train(**kwargs) -> torch.utils.data.Dataset:
    return DatasetTrain(cityscapes_data_path="datasets/cityscapes", cityscapes_meta_path="datasets/cityscapes/meta",
                        **kwargs)


def cityscapes_test(**kwargs) -> torch.utils.data.Dataset:
    return DatasetVal(cityscapes_data_path="datasets/cityscapes",
                      cityscapes_meta_path="datasets/cityscapes/meta", **kwargs)


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, **kwargs):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/train/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"
        self.transforms = train_transforms
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for train_dir in train_dirs:
            train_img_dir_path = self.img_dir + train_dir

            file_names = os.listdir(train_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = train_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "ground"
        ]

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)  # (shape: (1024, 2048))

        img = img[..., [2, 1, 0]]
        transformed = self.transforms(image=np.array(img), mask=label_img)
        image = transforms.ToTensor()(transformed['image'])
        seg = torch.from_numpy(transformed['mask'])

        return (image, seg.long())

    def __len__(self):
        return self.num_examples


class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, **kwargs):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/val/"
        self.label_dir = cityscapes_meta_path + "/label_imgs/"
        self.transforms = test_transforms
        self.reverse_normalization = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            torchvision.transforms.Normalize(std=[1, 1, 1], mean=[-0.485, -0.456, -0.406]),
        ])
        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        for val_dir in val_dirs:
            val_img_dir_path = self.img_dir + val_dir

            file_names = os.listdir(val_img_dir_path)
            for file_name in file_names:
                img_id = file_name.split("_leftImg8bit.png")[0]

                img_path = val_img_dir_path + file_name

                label_img_path = self.label_dir + img_id + ".png"
                label_img = cv2.imread(label_img_path, -1)  # (shape: (1024, 2048))

                example = {}
                example["img_path"] = img_path
                example["label_img_path"] = label_img_path
                example["img_id"] = img_id
                self.examples.append(example)

        self.num_examples = len(self.examples)
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "ground"
        ]

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)  # (shape: (1024, 2048))

        # convert numpy -> torch:
        img = img[..., [2, 1, 0]]
        transformed = self.transforms(image=np.array(img), mask=label_img)
        image = transforms.ToTensor()(transformed['image'])
        seg = torch.from_numpy(transformed['mask'])

        return (image, seg.long())

    def __len__(self):
        return self.num_examples

    @classmethod
    def reverse_augmentation(cls, data: torch.Tensor) -> torch.Tensor:
        var = torch.Tensor([0.229, 0.224, 0.225]).to(data)
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255).type(torch.uint8).detach().cpu()


class DatasetSeq(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path, sequence):
        self.img_dir = cityscapes_data_path + "/leftImg8bit/demoVideo/stuttgart_" + sequence + "/"

        self.img_h = 1024
        self.img_w = 2048

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split("_leftImg8bit.png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))
        # resize img without interpolation:
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples


class DatasetThnSeq(torch.utils.data.Dataset):
    def __init__(self, thn_data_path):
        self.img_dir = thn_data_path + "/"

        self.examples = []

        file_names = os.listdir(self.img_dir)
        for file_name in file_names:
            img_id = file_name.split(".png")[0]

            img_path = self.img_dir + file_name

            example = {}
            example["img_path"] = img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (512, 1024, 3))

        # normalize the img (with mean and std for the pretrained ResNet):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples


def label_img_to_color(img):
    label_to_color = {
        0: [128, 64, 128],  # road
        1: [244, 35, 232],  # sidewalk
        2: [70, 70, 70],  # building
        3: [102, 102, 156],  # wall
        4: [190, 153, 153],  # fence
        5: [153, 153, 153],  # pole
        6: [250, 170, 30],  # traffic light
        7: [220, 220, 0],  # traffic sign
        8: [107, 142, 35],  # vegetation
        9: [152, 251, 152],  # terrain
        10: [70, 130, 180],  # sky
        11: [220, 20, 60],  # person
        12: [255, 0, 0],  # rider
        13: [0, 0, 142],  # car
        14: [0, 0, 70],  # truck
        15: [0, 60, 100],  # bus
        16: [0, 80, 100],  # train
        17: [0, 0, 230],  # motorcycle
        18: [119, 11, 32],  # bicycle
        19: [81, 0, 81]  # ground
    }

    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])

    return img_color
