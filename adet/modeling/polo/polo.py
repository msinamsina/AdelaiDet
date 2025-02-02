# -*- coding: utf-8 -*-
import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from detectron2.layers import ShapeSpec, batched_nms, cat, paste_masks_in_image
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import sigmoid_focal_loss
import matplotlib.pyplot as plt

from .utils import imrescale, center_of_mass, point_nms, mask_nms, matrix_nms
from .loss import dice_loss, FocalLoss
from .lovasz_losses import LovaszSoftmax

__all__ = ["POLO"]


@META_ARCH_REGISTRY.register()
class POLO(nn.Module):
    """
    POLO model. Creates FPN backbone, instance branch for kernels and categories prediction,
    mask branch for unified mask features.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.POLO.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.POLO.FPN_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.POLO.SIGMA
        # Instance parameters.
        self.num_classes = cfg.MODEL.POLO.NUM_CLASSES
        self.num_kernels = cfg.MODEL.POLO.NUM_KERNELS
        self.num_grids = cfg.MODEL.POLO.NUM_GRIDS

        self.instance_in_features = cfg.MODEL.POLO.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.POLO.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.POLO.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.POLO.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.POLO.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.POLO.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.POLO.MASK_CHANNELS
        self.num_masks = cfg.MODEL.POLO.NUM_MASKS

        # Inference parameters.
        self.max_before_nms = cfg.MODEL.POLO.NMS_PRE
        self.score_threshold = cfg.MODEL.POLO.SCORE_THR
        self.update_threshold = cfg.MODEL.POLO.UPDATE_THR
        self.mask_threshold = cfg.MODEL.POLO.MASK_THR
        self.max_per_img = cfg.MODEL.POLO.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.POLO.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.POLO.NMS_SIGMA
        self.nms_type = cfg.MODEL.POLO.NMS_TYPE

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the ins head.
        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.ins_head = POLOInsHead(cfg, instance_shapes)

        # build the mask head.
        mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
        self.mask_head = POLOMaskHead(cfg, mask_shapes)

        # loss
        # self.ins_loss_weight = cfg.MODEL.POLO.LOSS.DICE_WEIGHT
        # self.focal_loss_alpha = cfg.MODEL.POLO.LOSS.FOCAL_ALPHA
        # self.focal_loss_gamma = cfg.MODEL.POLO.LOSS.FOCAL_GAMMA
        # self.focal_loss_weight = cfg.MODEL.POLO.LOSS.FOCAL_WEIGHT
        # self.seg_focal_loss_weight = cfg.MODEL.POLO.LOSS.SEG_FOCAL_WEIGHT
        self.obj_loss_weight = cfg.MODEL.POLO.LOSS.OBJ_WEIGHT
        self.seg_cross_loss_weight = cfg.MODEL.POLO.LOSS.SEGCROSS_WEIGHT
        self.seg_cross_loss_classes_weight = cfg.MODEL.POLO.LOSS.SEGCROSS_CLASSES_WEIGHT
        if self.seg_cross_loss_classes_weight is not None:
            self.seg_cross_loss_classes_weight = torch.tensor(self.seg_cross_loss_classes_weight).to(self.device)
        self.seg_lovasz_loss_weight = cfg.MODEL.POLO.LOSS.SEGLOVASZ_WEIGHT
        self.seg_lovasz_loss_perimg = cfg.MODEL.POLO.LOSS.SEGLOVASZ_PERIMG
        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.denorm = lambda x: ((x * pixel_std) + pixel_mean).to(torch.uint8)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
        Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        cate_pred, kernel_pred = self.ins_head(ins_features)

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        if self.training:
            """
            get_ground_truth.
            return loss and so on.
            """
            mask_feat_size = mask_pred.size()[-2:]
            targets = self.get_ground_truth(gt_instances, mask_feat_size)
            losses = self.loss(cate_pred, kernel_pred, mask_pred, targets)
            return losses
        else:
            # point nms.
            cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
            results = self.inference(cate_pred, kernel_pred, mask_pred, images.image_sizes, batched_inputs)
            return results


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for img_idx in range(len(gt_instances)):
            cur_ins_label_list, cur_cate_label_list, \
            cur_ins_ind_label_list, cur_grid_order_list = \
                self.get_ground_truth_single(img_idx, gt_instances,
                                             mask_feat_size=mask_feat_size)
            ins_label_list.append(cur_ins_label_list)
            cate_label_list.append(cur_cate_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def combine_part_of_parent_ins(self, gt_bboxes_raw, gt_labels_raw, gt_masks_raw, gt_parent_id_raw, device):
        ch = len(torch.unique(gt_parent_id_raw))
        w, h = gt_masks_raw.shape[1:]
        final_gt_masks_raw = torch.zeros((ch, w, h), dtype=torch.int, device=device)
        final_gt_labels_raw = torch.ones((ch,), dtype=torch.int, device=device)
        final_gt_bboxes_raw = torch.empty((ch, 4), dtype=torch.float32, device=device)
        for i in range(ch):
            flg = 1
            for idx, p_id in enumerate(gt_parent_id_raw):
                if p_id == i:
                    final_gt_masks_raw[i] = torch.add(final_gt_masks_raw[i], gt_masks_raw[idx] * gt_labels_raw[idx])
                    if flg:
                        final_gt_bboxes_raw[i] = gt_bboxes_raw[idx]
                        flg = 0
                    else:
                        final_gt_bboxes_raw[i, 0] = min(final_gt_bboxes_raw[i, 0], gt_bboxes_raw[idx, 0])
                        final_gt_bboxes_raw[i, 1] = min(final_gt_bboxes_raw[i, 1], gt_bboxes_raw[idx, 1])
                        final_gt_bboxes_raw[i, 2] = max(final_gt_bboxes_raw[i, 2], gt_bboxes_raw[idx, 2])
                        final_gt_bboxes_raw[i, 3] = max(final_gt_bboxes_raw[i, 3], gt_bboxes_raw[idx, 3])
            # import matplotlib.pyplot as plt
            # plt.figure(num=f'This is the title p id:{i}')
            # plt.imshow(final_gt_masks_raw[i].cpu().numpy())
            # print(i, final_gt_bboxes_raw[i])
        # plt.show()
        final_gt_masks_raw[final_gt_masks_raw >= self.num_classes] = 0
        return final_gt_bboxes_raw, final_gt_labels_raw, final_gt_masks_raw

    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
        gt_parent_id_raw = gt_instances[img_idx].parent_id
        device = gt_labels_raw[0].device

        gt_bboxes_raw, gt_labels_raw, gt_masks_raw = self.combine_part_of_parent_ins(gt_bboxes_raw, gt_labels_raw, gt_masks_raw, gt_parent_id_raw, device)
        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, 0) # only one class: person #self.num_classes
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_for_center_of_mass = torch.zeros(gt_masks.shape, dtype=torch.bool, device=device)
            gt_masks_for_center_of_mass[gt_masks > 0] = True
            center_ws, center_hs = center_of_mass(gt_masks_for_center_of_mass)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            gt_masks = gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            gt_masks = imrescale(gt_masks, scale=1./output_stride)

            if len(gt_masks.shape) == 2:
                gt_masks = gt_masks[..., None]
            gt_masks = torch.from_numpy(gt_masks).to(dtype=torch.uint8, device=device).permute(2, 0, 1)
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)
                # print(gt_label)
                cate_label[top:(down+1), left:(right+1)] = gt_label
                # print(cate_label.unique())
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)

        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def loss(self, cate_preds, kernel_preds, ins_pred, targets):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = targets
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):
                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]

                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = torch.reshape(cur_ins_pred, (-1, self.num_classes, H, W))
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1, 1)
                cur_ins_pred = F.conv3d(cur_ins_pred, kernel_pred, stride=1).view(-1, self.num_classes, H, W)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)
        # dice loss
        loss_ins = []
        lovasz_loss_ins = []
        lovosz_softmax = LovaszSoftmax(ignore_index=0)
        # focal_loss_ins = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue

                 # F.cross_entropy(input, target.to(torch.long), reduction='mean',ignore_index=0)
            loss_ins.append(F.cross_entropy(input, target.to(torch.long), reduction='mean', weight=self.seg_cross_loss_classes_weight))
            # loss_ins.append(F.cross_entropy(input, target.to(torch.long), reduction='mean', weight=self.seg_cross_loss_classes_weight))
            # lovasz_loss_ins.append(lovosz_softmax(input, target.to(torch.long)))
            lovasz_loss_ins.append(F.cross_entropy(input, target.to(torch.long), reduction='mean',ignore_index=0))
            # input = torch.sigmoid(input)
            # loss_ins.append(dice_loss(input, target))
            # target img to one-hot
            # target = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()
            # flatten input and target
            # input = input.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            # target = target.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

            # focal_loss_ins.append(sigmoid_focal_loss(input, target,
            #                         gamma=2.0,
            #                         alpha=0.25,
            #                         reduction="mean"))
            # weight=torch.tensor([10.0, 100, 10, 100, 100, 10, 10, 10, 100, 10, 100, 100, 100, 100, 10, 100, 100, 100, 100, 100]).to(self.device)/100))
            # input = F.softmax(input, dim=1)
            # lovasz_loss_ins.append(lovasz_softmax(input, target, ignore=255, per_image=self.seg_lovasz_loss_perimg, classes='all'))

        # loss_ins_mean = torch.cat(loss_ins).mean()
        loss_ins_mean = torch.stack(loss_ins).mean()
        # loss_focal_ins_mean = torch.stack(focal_loss_ins).mean()
        lovasz_loss_ins_mean = torch.stack(lovasz_loss_ins).mean()
        loss_seg_cross = loss_ins_mean * self.seg_cross_loss_weight
        loss_seg_lovasz = lovasz_loss_ins_mean * self.seg_lovasz_loss_weight
        # loss_seg_focal = self.seg_focal_loss_weight * loss_focal_ins_mean


        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, 1)  #  self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)
        # prepare one_hot
        # pos_inds = torch.nonzero(flatten_cate_label != self.num_classes).squeeze(1)
        # pos_inds = torch.nonzero(flatten_cate_labels != 1).squeeze(1)
        #
        # flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        # flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1
        # loss_cate = self.focal_loss_weight * sigmoid_focal_loss_jit(flatten_cate_preds, flatten_cate_labels_oh,
        #                             gamma=self.focal_loss_gamma,
        #                             alpha=self.focal_loss_alpha,
        #                             reduction="sum") / (num_ins + 1)

        loss_object = self.obj_loss_weight * \
                    F.binary_cross_entropy_with_logits(
                        flatten_cate_preds.squeeze(), flatten_cate_labels.to(torch.float32), reduction='sum') / (num_ins + 1)

        return {
                'loss_seg_cross': loss_seg_cross,
                # 'loss_seg_lovasz': loss_seg_lovasz,
                'loss_seg_class': loss_seg_lovasz,
                # 'loss_seg_focal': loss_seg_focal,
                'loss_object': loss_object}

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))


    def inference(self, pred_cates, pred_kernels, pred_masks, cur_sizes, images):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, 1).detach() #self.num_classes).detach()
                          for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_cate, pred_kernel, pred_mask,
                                                 cur_sizes[img_idx], ori_size)
            results.append({"instances": result})
        return results

    def inference_single_image(
            self, cate_preds, kernel_preds, seg_preds, cur_size, ori_size
    ):
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = math.ceil(h/f_h)
        upsampled_size_out = (int(f_h*ratio), int(f_w*ratio))
        # sort and keep top_k
        sorted, sort_inds = torch.sort(cate_preds, descending=True, dim=0)
        sort_inds = sort_inds[0:1000]
        cate_preds = cate_preds[sort_inds]
        kernel_preds = kernel_preds[sort_inds[:, 0]]
        inds = cate_preds > self.score_threshold
        cate_scores = cate_preds[inds]

        if len(cate_scores) == 0:
            # print('no cate_scores')
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        H, W = seg_preds.shape[-2:]
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1, 1)
        seg_preds = torch.reshape(seg_preds, (-1, self.num_classes, H, W))
        seg_preds = seg_preds.unsqueeze(0)
        # print('seg_preds', seg_preds.shape)
        # print('kernel_preds', kernel_preds.shape)
        # raise ''
        seg_preds = F.conv3d(seg_preds, kernel_preds, stride=1).view(-1, self.num_classes, H, W)
        # mask.
        seg_masks = torch.argmax(seg_preds, dim=1)
        seg_masks_list = []
        keep_inds = []
        for i, m in enumerate(seg_masks):
            if m.unique().shape[0] > 2:
                seg_masks_list.append(m)
                keep_inds.append(i)

        if len(seg_masks_list) == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results


        seg_masks = torch.stack(seg_masks_list, dim=0)
        keep_inds = torch.tensor(keep_inds)
        cate_scores = cate_scores[keep_inds]
        cate_labels = cate_labels[keep_inds]
        strides = strides[keep_inds]

        binary_masks = seg_masks.clone()
        binary_masks[binary_masks > 0] = 1
        sum_masks = binary_masks.sum((1, 2)).float()
        # cate_scores = cate_scores * sum_masks / (h * w)
        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_masks = seg_masks[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        # seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        # cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        binary_masks = binary_masks[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]
        #
        if self.nms_type == "matrix":
            # matrix nms & filter.
            for i in range(2):
                cate_scores = matrix_nms(cate_labels, binary_masks, sum_masks, cate_scores,
                                              sigma=self.nms_sigma, kernel=self.nms_kernel)
                cate_scores[cate_scores.isnan()] = 0

            keep = cate_scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, binary_masks, sum_masks, cate_scores,
                                 nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        binary_masks = binary_masks[keep, :, :]
        seg_masks = seg_masks[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        binary_masks = binary_masks[sort_inds, :, :]
        seg_masks = seg_masks[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        # seg_preds = F.interpolate(seg_preds.unsqueeze(0),
        #                           size=upsampled_size_out,
        #                           mode='bilinear')[:, :, :h, :w]

        # seg_masks = F.interpolate(seg_preds,
        #                size=ori_size,
        #                           mode='bilinear').squeeze(0)
        # seg_masks = seg_masks > self.mask_threshold

        seg_masks = F.interpolate(seg_masks.unsqueeze(0).float(),
                                  size=ori_size,
                                  mode='nearest').squeeze(0)

        results = Instances(ori_size)
        results.pred_classes = cate_labels
        results.scores = cate_scores
        results.pred_masks = seg_masks

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)
        #for i in range(seg_masks.size(0)):
        #    mask = seg_masks[i].squeeze()
        #    ys, xs = torch.where(mask)
        #    pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        results.pred_boxes = Boxes(pred_boxes)

        return results


class POLOInsHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        POLO Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.POLO.NUM_CLASSES
        self.num_kernels = cfg.MODEL.POLO.NUM_KERNELS
        self.num_grids = cfg.MODEL.POLO.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.POLO.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.POLO.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.POLO.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.POLO.INSTANCE_CHANNELS
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.POLO.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"cate": (cfg.MODEL.POLO.NUM_INSTANCE_CONVS,
                                 cfg.MODEL.POLO.USE_DCN_IN_INSTANCE,
                                 False),
                        "kernel": (cfg.MODEL.POLO.NUM_INSTANCE_CONVS,
                                   cfg.MODEL.POLO.USE_DCN_IN_INSTANCE,
                                   cfg.MODEL.POLO.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.POLO.NORM == "none" else cfg.MODEL.POLO.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.POLO.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                        chn, self.instance_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, 1,  # self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.cate_tower, self.kernel_tower,
            self.cate_pred, self.kernel_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.POLO.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        cate_pred = []
        kernel_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))
            # cate_pred.append(torch.nn.functional.sigmoid(self.cate_pred(cate_feat)))
        return cate_pred, kernel_pred


class POLOMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        POLO Mask Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.POLO.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.POLO.NUM_MASKS
        self.mask_in_features = cfg.MODEL.POLO.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.POLO.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.POLO.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        # fmt: on
        norm = None if cfg.MODEL.POLO.NORM == "none" else cfg.MODEL.POLO.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks*self.num_classes,
                kernel_size=1, stride=1,
                padding=1, bias=norm is None),
            # nn.GroupNorm(self.num_classes, self.num_masks*self.num_classes),
            nn.ReLU(inplace=True),
            # nn.Conv2d(
            #     self.num_masks * 4, self.num_masks * self.num_classes,
            #     kernel_size=3, stride=1,
            #     padding=1, bias=norm is None),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(
            #     self.num_masks*4, self.num_masks*self.num_classes,
            #     kernel_size=1, stride=1,
            #     padding=0, bias=norm is None),
            # nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)
        return mask_pred
