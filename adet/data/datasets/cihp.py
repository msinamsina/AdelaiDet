import os
from detectron2.structures import BoxMode
from imantics import Polygons, Mask
import pickle
import cv2
import numpy as np


def get_cihp_dicts(root, train=False, person_only=False):
    """
    Args:
        root (str): the directory where the dataset is stored
        train (bool): whether to load the training or the test set
        person_only (bool): whether to only load the person class

    Returns:
        list[dict]: a list of dicts in detectron2's standard format

    """
    if train:
        anno = os.path.join(root, 'CIHP/Training/train_id.txt')
        root = os.path.join(root, 'CIHP/Training/')
    else:
        anno = os.path.join(root, 'CIHP/Validation/val_id.txt')
        root = os.path.join(root, 'CIHP/Validation/')

    with open(anno, 'r') as f:
        anno_ids = f.read().strip().split('\n')

    try:
        if person_only:
            with open(os.path.join(root, 'person_dataset.pk'), 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            with open(os.path.join(root, 'dataset.pk'), 'rb') as f:
                data = pickle.load(f)
            return data
    except FileNotFoundError:
        dataset_dicts = []
        for idx, id in enumerate(anno_ids):
            if id == '0035374':
                continue
            record = {}
            filename = os.path.join(root, 'Images', id + '.jpg')
            print(filename)
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            if not person_only:
                inst_filename = os.path.join(root, 'Instance_ids', id + '.png')
                inst_img = cv2.imread(inst_filename, 0)
                inst_img[inst_img == 255] = 0
                instances = np.unique(inst_img)
                for inst in instances:
                    if inst == 0:
                        continue
                    mask = inst_img.copy()
                    mask[mask != inst] = 0
                    mask[mask == inst] = 1

                    polygons = Mask(mask).polygons()
                    xy = polygons.bbox()

                    poly = polygons.segmentation

                    # filter out small polygons
                    true_polygons_list = []
                    for p in poly:
                        if len(p) > 5:
                            true_polygons_list.append(p)

                    if len(true_polygons_list) < 1:
                        continue

                    obj = {
                        "bbox": list(xy),
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": true_polygons_list,
                        "category_id": (inst % 20),
                        "person_id": (inst // 20),
                    }
                    if obj['category_id'] < 0:
                        print(obj)
                        print(instances)
                    objs.append(obj)

            h_inst_filename = os.path.join(root, 'Human_ids', id + '.png')
            h_inst_img = cv2.imread(h_inst_filename, 0)
            h_inst_img[h_inst_img == 255] = 0

            instances = np.unique(h_inst_img)
            for inst in instances:
                if inst == 0:
                    continue
                mask = h_inst_img.copy()
                mask[mask != inst] = 0
                mask[mask == inst] = 1

                polygons = Mask(mask).polygons()
                xy = polygons.bbox()

                poly = polygons.segmentation
                true_polygons_list = []
                for p in poly:
                    if len(p) > 5:
                        true_polygons_list.append(p)

                if len(true_polygons_list) < 1:
                    continue
                obj = {
                    "bbox": list(xy),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": true_polygons_list,
                    "category_id": 0,
                    "person_id": inst-1
                }
                if obj['category_id'] < 0:
                    print(obj)
                    print(instances)
                objs.append(obj)

            record["annotations"] = objs
            record['sem_seg_file_name'] = os.path.join(root, 'Category_ids', id + '.png')
            dataset_dicts.append(record)
        if person_only:
            with open(os.path.join(root, 'person_dataset.pk'), 'wb') as f:
                pickle.dump(dataset_dicts, f)
        else:
            with open(os.path.join(root, 'dataset.pk'), 'wb') as f:
                pickle.dump(dataset_dicts, f)

    return dataset_dicts
