import os.path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator
import matplotlib.pyplot as plt
import torch

import numpy as np
from PIL import Image, ImageDraw
import time
from .utils import poly_to_mask, plot_mask, voc_ap, cal_one_mean_iou


class APEvaluator:

    def __init__(self):
        self.tp = []
        self.fp = []

        self.precision = []
        self.recall = []
        self.ap = []

    def add_tp(self):
        self.tp.append(1)
        self.fp.append(0)

    def add_fp(self):
        self.tp.append(0)
        self.fp.append(1)

    def eval(self, npos):
        tp = np.array(self.tp)
        fp = np.array(self.fp)
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / npos
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = voc_ap(rec, prec)
        self.precision = prec
        self.recall = rec
        self.ap = ap
        return ap




class MHPDatasetEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__()
        self._cfg = cfg.clone()
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_dicts = DatasetCatalog.get(dataset_name)
        self.metadata = MetadataCatalog.get(dataset_name)
        self.num_classes = len(self.metadata.thing_classes)
        self.ovthresh_seg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            pass

    def reset(self):
        self.apr = {}
        self.app = {}
        for i in self.ovthresh_seg:
            self.app[i] = APEvaluator()
            self.apr[i] = APEvaluator()
        self.npos = 0
        self.npart = 0
        self.total_time = 0
        self.delta_time = time.time()
        self.num_images = 0

    def process(self, inputs, outputs):
        self.num_images += len(inputs)
        self.total_time += (time.time() - self.delta_time)
        for input, output in zip(inputs, outputs):
            if len(output["instances"]) == 0:
                seg_gt = self.mix_parts_of_instance(self.dataset_dicts[input['image_id']]['annotations'], (100, 100))
                self.npos += seg_gt.shape[0]
                for i in range(seg_gt.shape[0]):
                    self.npart += len(np.unique(seg_gt[i]))
                continue
            w, h = output["instances"].pred_masks.size(1), output["instances"].pred_masks.size(2)
            seg_gt = self.mix_parts_of_instance(self.dataset_dicts[input['image_id']]['annotations'], (w, h))
            self.npos += seg_gt.size(0)
            for i in range(seg_gt.size(0)):
                self.npart += len(np.unique(seg_gt[i]))

            seg_pred = output["instances"].pred_masks

            list_mious = []
            list_ious = []
            for i in range(seg_pred.size(0)):
                max_miou = 0
                max_iou = []
                max_iou_id = -1
                a = seg_pred[i].clone().to('cpu')
                for j in range(seg_gt.size(0)):
                    b = seg_gt[j].clone().to('cpu')
                    b[b >= self.num_classes] = 0

                    seg_iou = cal_one_mean_iou(a.numpy().astype(np.uint8), b.numpy().astype(np.uint8), 7)
                    # print(seg_iou)
                    # seg_iou = seg_iou[b.unique().cpu().numpy().astype(np.uint8)]
                    # seg_iou[seg_iou == 0] = np.nan
                    mean_seg_iou = np.nanmean(seg_iou[0:])
                    # print(mean_seg_iou)
                    if mean_seg_iou > max_miou:
                        max_miou = mean_seg_iou
                        max_iou = seg_iou
                        max_iou_id = j
                # print(len(max_iou))
                list_mious.append({"id": max_iou_id, "iou": max_miou, "iou_list": max_iou})

            list_mious = sorted(list_mious, key=lambda x: x["iou"], reverse=True)
            # print([f"{x['id']}:{x['iou']:.3f}" for x in list_mious])
            for j in self.ovthresh_seg:
                id_list = []
                for i in list_mious:
                    if i['id'] not in id_list:
                        # print("aa", len(i['iou_list']))
                        for k in range(len(i['iou_list'])):
                            if i['iou_list'][k] == np.nan:
                                continue
                            if i['iou_list'][k] >= j:
                                self.apr[j].add_tp()
                            else:
                                self.apr[j].add_fp()
                        if i["iou"] >= j:
                            id_list.append(i['id'])
                            self.app[j].add_tp()
                        else:
                            self.app[j].add_fp()
                    else:
                        self.app[j].add_fp()


            # plot_mask(seg_gt, self.dataset_dicts.colormap, 20, 2, os.path.join(self._output_dir, str(input['image_id']) + "_gt.png"))
            # plot_mask(seg_pred, self.dataset_dicts.colormap, 20, 2, os.path.join(self._output_dir, str(input['image_id']) + "_pred.png"))
            # img = input["image"].permute(1, 2, 0).cpu().numpy()
            # img = (img * 255).astype(np.uint8)
            # Image.fromarray(img).save(os.path.join(self._output_dir, str(input['image_id']) + ".png"))
            # plt.show()
        # self.evaluate()
        self.delta_time = time.time()
        # return self.evaluate()

    def mix_parts_of_instance(self, instances, size):
        person_ids = set()
        for i in instances:
            person_ids.add(i['parent_id'])

        h, w = size
        seg_mask = torch.zeros((len(person_ids), h, w))
        # print(person_ids)
        for i in person_ids:
            for j in instances:
                if j['parent_id'] == i:
                    mask = poly_to_mask(j['segmentation'], w, h)
                    mask = torch.from_numpy(mask)
                    seg_mask[i] = torch.add(seg_mask[i], mask * (j['category_id'] + 0))

        return seg_mask

    def evaluate(self):
        result = {}
        app = []
        apr = []
        for i in self.ovthresh_seg:

            result[f"APr_{i}"] = self.apr[i].eval(self.npart)
            result[f"APp_{i}"] = self.app[i].eval(self.npos)
            print(f"APr_{i} = {result[f'APr_{i}']:.3f}")
            print(f"APp_{i} = {result[f'APp_{i}']:.3f}")
            app.append(result[f"APp_{i}"])
            apr.append(result[f"APr_{i}"])

            # tp = np.array(self.tp[i])
            # fp = np.array(self.fp[i])
            # tp = np.cumsum(tp)
            # fp = np.cumsum(fp)
            # rec = tp / self.npos
            # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            #
            # ap = voc_ap(rec, prec)
            # print(f"APp@{i}: {ap:.3f}, {self.npos}, {tp[-1]}, {fp[-1]}")
            # result[f"APp@{i}"] = ap

        #result["APpvol"] = sum(result.values()) / len(result)
        result["APpvol"] = sum(app) / len(app)
        result["APrvol"] = sum(apr) / len(apr)
        result["total_time"] = self.total_time
        result["fps"] = self.num_images / self.total_time
        # print(f"APpvol: {result['APpvol']:.3f}")
        print(f"total_time: {result['total_time']:.2f}")
        print(f"fps: {result['fps']:.2f}")
        return result
