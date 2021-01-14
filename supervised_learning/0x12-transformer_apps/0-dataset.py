#!/usr/bin/env python3
""" doc """

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf


class Dataset:
    """ doc """
    def __init__(self):
        """ doc """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """ doc """
        token_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        token_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return token_pt, token_en
