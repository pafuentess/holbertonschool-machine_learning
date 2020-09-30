#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K
import numpy as np


class Yolo:
    """ doc """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ doc """
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        self.class_names = []

        with open(classes_path, 'r') as f:
            for line in f:
                self.class_names.append(line.strip())
            self.class_names.pop()

    def sigmoid(self, x):
        """ doc """
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """ doc """
        image_h = image_size[0]
        image_w = image_size[1]
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        for i, box in enumerate(boxes):
            grid_H = box.shape[0]
            grid_W = box.shape[1]
            anchor_boxes = box.shape[2]

            caja = np.zeros((grid_H, grid_W, anchor_boxes))

            Index_y = np.arange(grid_H)
            Index_x = np.arange(grid_W)

            Index_y = Index_y.reshape(grid_H, 1, 1)
            Index_x = Index_x.reshape(grid_W, 1, 1)

            caja_y = caja + Index_y
            caja_x = caja + Index_x

            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            n_tx = self.sigmoid(t_x)
            n_ty = self.sigmoid(t_y)

            bx = n_tx + caja_x
            by = n_ty + caja_y

            bx /= grid_W
            by /= grid_H

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            Input_w = self.model.input.shape[1].value
            Input_h = self.model.input.shape[2].value

            bw /= Input_w
            bh /= Input_h

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            box[..., 0] = x1 * image_w
            box[..., 1] = y1 * image_h
            box[..., 2] = x2 * image_w
            box[..., 3] = y2 * image_h

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ doc """
        scores = []
        for conf, prob in zip(box_confidences, box_class_probs):
            scores.append(conf * prob)

        box_class_scores = [score.max(axis=3) for score in scores]
        print(box_class_scores)
        box_class_scores = [score.reshape(-1) for score in box_class_scores]
        print(box_class_scores)
        box_score = np.concatenate(box_class_scores)

        del_index = np.where(box_score < self.class_t)

        box_score = np.delete(box_score, del_index)

        box_class_list = [box.argmax(axis=3) for box in scores]
        box_class_list = [box.reshape(-1) for box in box_class_list]
        box_classes = np.concatenate(box_class_list)
        box_classes = np.delete(box_classes, del_index)

        box_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(box_list, axis=0)
        filtered_boxes = np.delete(boxes, del_index, axis=0)

        return filtered_boxes, box_classes, box_score
