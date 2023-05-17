from torch.utils.data import ConcatDataset

from models.SSD.ssd.config.path_catlog import DatasetCatalog
from models.SSD.ssd.data.datasets.coco import COCODataset
from models.SSD.ssd.data.datasets.voc import VOCDataset


def build_dataset(dataset, dataset_list, path="/home/fe/dreyer/projects/coco/data", transform=None, target_transform=None, is_train=True, preprocessing=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name)
        args = data['args']
        factory = dataset
        args['preprocessing'] = preprocessing
        args['transform'] = transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = not is_train
        elif factory == COCODataset:
            args['remove_empty'] = is_train
        args['data_dir'] = path + args['data_dir'].replace("datasets", "")
        if "ann_file" in args.keys():
            args['ann_file'] = path + args['ann_file'].replace("datasets", "")
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
