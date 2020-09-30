#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


class Yolo:
    """ doc """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ doc """
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
