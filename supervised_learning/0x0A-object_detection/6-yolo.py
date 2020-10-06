#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


class Yolo:
    """ doc """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor of the class
        Arguments:
         - model_path is the path to where a Darknet Keras model is stored
         - classes_path is the path to where the list of class names used for
            the Darknet model, listed in order of idx, can be found
         - class_t is a float representing the box score threshold for
            the initial filtering step
         - nms_t is a float representing the IOU threshold for
            non-max suppression
         - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            * outputs is the number of outputs (predictions) made by
                the Darknet model
            * anchor_boxes is the number of anchor boxes used for
                each prediction
            * 2 => [anchor_box_width, anchor_box_height]
        Public instance attributes:
         - model: the Darknet Keras model
         - class_names: a list of the class names for the model
         - class_t: the box score threshold for the initial filtering step
         - nms_t: the IOU threshold for non-max suppression
         - anchors: the anchor boxes
        """

        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

    def sigmoid(self, x):
        """
        Function that calculates sigmoid
        """
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Public method to process the outputs
        Arguments:
         - outputs is a list of numpy.ndarrays containing the predictions
            from the Darknet model for a single image:
            Each output will have the shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
             * grid_height & grid_width => the height and width of
                the grid used for the output
             * anchor_boxes => the number of anchor boxes used
             * 4 => (t_x, t_y, t_w, t_h)
             * 1 => box_confidence
             * classes => class probabilities for all classes
         - image_size is a numpy.ndarray containing the image’s original size
            [image_size[0], image_size[1]]
        Returns:
         A tuple of (boxes, box_confidences, box_class_probs):
         - boxes: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output:
            * 4 => (x1, y1, x2, y2)
            * (x1, y1, x2, y2) should represent the boundary box
                relative to original image
         - box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1)
            containing the box confidences for each output, respectively
         - box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes)
            containing the box’s class probabilities
            for each output, respectively
        """
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
            Index_x = Index_x.reshape(1, grid_W, 1)

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
        """
        Public method to filter the boxes
        Arguments:
         - boxes: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 4)
            containing the processed boundary boxes for each output
         - box_confidences: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, 1)
            containing the processed box confidences for each output
         - box_class_probs: a list of numpy.ndarrays of shape
             (grid_height, grid_width, anchor_boxes, classes)
            containing the processed box class probabilities for each output
        Returns:
         A tuple of (filtered_boxes, box_classes, box_scores):
         * filtered_boxes: a numpy.ndarray of shape (?, 4) containing
            all of the filtered bounding boxes:
         * box_classes: a numpy.ndarray of shape (?,) containing
            the class number that each box in filtered_boxes predicts,
            respectively
         * box_scores: a numpy.ndarray of shape (?) containing
            the box scores for each box in filtered_boxes, respectively
        """
        scores = []
        for conf, prob in zip(box_confidences, box_class_probs):
            scores.append(conf * prob)

        box_class_scores = [score.max(axis=3) for score in scores]
        box_class_scores = [score.reshape(-1) for score in box_class_scores]
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

    @staticmethod
    def iou(boxA, boxB):
        """
        Method to calculate intersection over union
        (x1, y1, x2, y2)
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(xB - xA, 0) * max(yB - yA, 0)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Public method to select the boxes to keep by its score idx
        Arguments:
         - filtered_boxes: a numpy.ndarray of shape (?, 4)
            containing all of the filtered bounding boxes:
         - box_classes: a numpy.ndarray of shape (?,)
            containing the class number for the class that filtered_boxes
            predicts, respectively
         - box_scores: a numpy.ndarray of shape (?)
            containing the box scores for each box in filtered_boxes,
            respectively
        Returns:
         A tuple of
           (box_predictions, predicted_box_classes, predicted_box_scores):
         - box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
         - predicted_box_classes: a numpy.ndarray of shape (?,)
            containing the class number for box_predictions
            ordered by class and box score, respectively
         - predicted_box_scores: a numpy.ndarray of shape (?)
            containing the box scores for box_predictions
            ordered by class and box score, respectively
        """
        index = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in index])
        predict_box_classes = np.array([box_classes[i] for i in index])
        predict_box_scores = np.array([box_scores[i] for i in index])

        _, number_counts = np.unique(predict_box_classes,
                                     return_counts=True)

        i = 0
        acummulated = 0

        for number_count in number_counts:
            while i < acummulated + number_count:
                j = i + 1
                while j < acummulated + number_count:
                    temp = self.iou(box_predictions[i], box_predictions[j])
                    if temp > self.nms_t:
                        box_predictions = np.delete(box_predictions,
                                                    j, axis=0)
                        predict_box_scores = np.delete(predict_box_scores,
                                                       j, axis=0)
                        predict_box_classes = np.delete(predict_box_classes,
                                                        j, axis=0)
                        number_count -= 1
                    else:
                        j += 1
                i += 1
            acummulated += number_count

        return box_predictions, predict_box_classes, predict_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Static method to load the images form a path
        Arguments:
         - folder_path: a string representing the path to the folder
            holding all the images to load
        Returns:
         A tuple of (images, image_paths):
             * images: a list of images as numpy.ndarrays
             * image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(folder_path + '/*')
        image = [cv2.imread(image) for image in image_paths]

        return image, image_paths

    def preprocess_images(self, images):
        """
        Public method to process images
        Arguments:
         - images: a list of images as numpy.ndarrays
        Returns:
         A tuple of (pimages, image_shapes):
            - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all of the preprocessed images
                * ni: the number of images that were preprocessed
                * input_h: the input height for the Darknet model
                * input_w: the input width for the Darknet model
                * 3: number of color channels
            - image_shapes: a numpy.ndarray of shape (ni, 2)
                containing the original height and width of the images
                * 2 => (image_height, image_width)
        """
        Input_w = self.model.input.shape[1].value
        Input_h = self.model.input.shape[2].value

        list_images = []
        i_shapes_list = []

        for image in images:
            img_shape = image.shape[0], image.shape[1]
            i_shapes_list.append(img_shape)

            resized = cv2.resize(image, (Input_w, Input_h),
                                 interpolation=cv2.INTER_CUBIC)

            pimage = resized / 255
            list_images.append(pimage)

        pimages = np.array(list_images)
        image_shapes = np.array(i_shapes_list)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Public method that displays a image with all boundary boxes,
        class names and box scores.
        Arguments:
         - image: a numpy.ndarray containing an unprocessed image
         - boxes: a numpy.ndarray containing the boundary boxes for the image
         - box_classes: a numpy.ndarray containing the class indices
            for each box
         - box_scores: a numpy.ndarray containing the box scores for each box
         - file_name: the file path where the original image is stored
            If the s key is pressed:
                The image should be saved in the directory detections,
                located in the current directory
            If any key besides s is pressed, the image window should be
            closed without saving
        """
        for i in range(len(boxes)):
            score = "{:.2f}".format(box_scores[i])

            init_point = (int(boxes[i, 0]), int(boxes[i, 1]))
            end_ponint = (int(boxes[i, 2]), int(boxes[i, 3]))

            color = (255, 0, 0)

            thickness = 2

            image = cv2.rectangle(image, init_point,
                                  end_ponint, color,
                                  thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(boxes[i, 0]), int(boxes[i, 1] - 5))
            fontscale = 0.5

            color = (0, 0, 255)
            thickness = 1

            image = cv2.putText(image,
                                self.class_names[box_classes[i]] + score,
                                org, font, fontscale, color, thickness,
                                cv2.LINE_AA)

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)

        if key == ord('s'):
            os.mkdir('detections') if not os.path.isdir('detections') else None
            os.chdir('detections')
            cv2.imwrite(file_name, image)
            os.chdir('../')

        cv2.destroyAllWindows()
