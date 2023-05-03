from typing import Dict, Any

from datasets.coco2017 import coco2017_train, coco2017_test
from datasets.cityscapes import cityscapes_train, cityscapes_test
from datasets.voc2012 import voc2012_train, voc2012_test

DATASETS = {
    "coco2017":
        {"train": coco2017_train,
         "test": coco2017_test,
         "n_classes": 80},
    "cityscapes": 
        {"train": cityscapes_train,
         "test": cityscapes_test,
         "n_classes": 20},
    "voc2012":
        {"train": voc2012_train,
         "test": voc2012_test,
         "n_classes": 21},
}


def get_dataset(dataset_name: str) -> Dict[str, Any]:
    print("Initialize dataset:", dataset_name)
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        return dataset
    else:
        print(f"DATASET {dataset_name} not defined.")
        exit()
