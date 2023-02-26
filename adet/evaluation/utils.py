import os.path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator
import matplotlib.pyplot as plt
import torch

import numpy as np
from PIL import Image, ImageDraw
import time



def poly_to_mask(polygon, width, height):
    img = Image.new('L', (width, height), 0)
    for poly in polygon:
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    mask = np.array(img)
    return mask


def seg_masks_to_rgb_img(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def plot_mask(mask, colormap, classes=20, row=1, mask_name=None):
    col = ((mask.size(0)) // row) + 2
    fig, ax = plt.subplots(col, row, figsize=(10, 10))
    for i in range(mask.size(0)):
        mask[mask >= 7 ] = 0
        prediction_colormap = seg_masks_to_rgb_img(mask[i].squeeze().cpu().numpy(), colormap, classes)
        #save the mask
        if mask_name is not None:
            Image.fromarray(prediction_colormap).save(mask_name+'_'+str(i)+'.png')
        ax[i // row, i % row].imshow(prediction_colormap)
    if mask_name is not None:
        plt.savefig(mask_name)


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / union
    return iu
