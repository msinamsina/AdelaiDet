import os
from detectron2.structures import BoxMode
from imantics import Polygons, Mask
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from matplotlib import pyplot as plt


class PPPDataset(Dataset):
    def __init__(self, root, train=False):
        self.train = train

        # Loading the Colormap
        colormap = loadmat(os.path.join(root, 'CIHP/human_colormap.mat')
        )["colormap"]
        colormap = colormap * 100
        self.colormap = colormap.astype(np.uint8)
        self.root = os.path.join(root, 'VOCdevkit/VOC2010/')
        if train:
            dataset = "train_id.txt"
        else:
            dataset = "val_id.txt"

        l = None
        with open(os.path.join(self.root, 'pascal_person_part/pascal_person_part_trainval_list', dataset)) as f:
            self.anno_ids = f.read()
            self.anno_ids = self.anno_ids.split('\n')[:-1]
            # try:
            #     self.anno_ids.remove('2009_003166')
            #     self.anno_ids.remove('2008_000572')
            #     self.anno_ids.remove('2009_005085')
            #     self.anno_ids.remove('2008_000008')
            #     self.anno_ids.remove('2008_000036')
            # except:
            #     pass


    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        pictur_id = self.anno_ids[idx]

        record = {}
        filename = os.path.join(self.root, 'JPEGImages', pictur_id + '.jpg')
        # print(f'file name: {filename}')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["annotations"] = self.create_annotations(pictur_id)
        record['sem_seg_file_name'] = os.path.join(self.root, "pascal_person_part/pascal_person_part_gt", pictur_id + '.png')
        return record

    def create_annotations(self, pictur_id):
        part_anno = loadmat(os.path.join(self.root, "Annotations_Part", pictur_id + '.mat'))
        person_part_mask = self.read_mask(os.path.join(self.root, "pascal_person_part/pascal_person_part_gt", pictur_id + '.png'))

        # plt.imshow(person_part_mask)
        # plt.show()

        inst_img = None
        cnt = 0
        objs = []
        for i in range(len(part_anno['anno'][0, 0][1][0])):
            # print(part_anno['anno'][0, 0][1][0, i][0][0])
            if part_anno['anno'][0, 0][1][0, i][0][0] == 'person':
                inst_img = part_anno['anno'][0, 0][1][0, i][2] * person_part_mask

                # plt.imshow(inst_img)
                # plt.show()
                flg = False
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
                        "category_id": inst ,
                        "parent_id": cnt,
                    }
                    if obj['category_id'] < 0:
                        print(obj)
                        print(instances)
                    objs.append(obj)
                    flg = True
                # print(np.unique(inst_img))
                if flg:
                    cnt += 1
                # cnt += 1
        return objs

    def __len__(self):
        # return 1
        return len(self.anno_ids)

    def get_dicts(self):
        return [self.__getitem__(i) for i in range(len(self))]

    def read_mask(self, filename):
        mask = cv2.imread(filename)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask[mask == 255] = 0
        mask[mask == 15] = 4
        mask[mask == 38] = 1
        mask[mask == 53] = 5
        mask[mask == 75] = 2
        mask[mask == 90] = 6
        mask[mask == 113] = 3
        return mask


if __name__ == '__main__':
    dataset = PPPDataset('/media/aras_vision/SSD/sina/Other-src/datasets/', train=True)
    for i in range(len(dataset)):
        print(dataset[i])
