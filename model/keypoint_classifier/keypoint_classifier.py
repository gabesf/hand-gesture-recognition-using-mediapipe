#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        converted_value = np.squeeze(result)

        result_index = np.argmax(converted_value)
        # print()
        # print(result_index)
        # for value in np.squeeze(result):
        #     print(f'{value:.20f}')

        if result_index == 1:
            if converted_value[result_index] < 0.95:
                result_index = 8

        if result_index == 2:
            if converted_value[result_index] < 0.95:
                result_index = 8

        if result_index == 3:
            if converted_value[result_index] < 0.9:
                result_index = 8

        if result_index == 4:
            if converted_value[result_index] < 0.8:
                result_index = 8

        if result_index == 5:
            if converted_value[result_index] < 0.8:
                result_index = 8

        if result_index == 6:
            if converted_value[result_index] < 0.8:
                result_index = 8

        if result_index == 7:
            if converted_value[result_index] < 0.8:
                result_index = 8
        return result_index
