#!/usr/bin/env python3
""" doc """

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf


class Dataset:
    """ doc """
    def __init__(self, batch_size, max_len):
        """ doc """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_length=max_len):
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        self.metadata = metadata

        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()

        size_trainData = self.metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(size_trainData)
        padded = ([None], [None])
        self.data_train.padded_batch(batch_size, padded_shapes=padded)
        self.data_train = \
            self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid.padded_batch(batch_size, padded_shapes=padded)

    def tokenize_dataset(self, data):
        """ doc """
        token_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        token_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return token_pt, token_en

    def encode(self, pt, en):
        """ doc """
        TokensEn = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        TokensPt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        return TokensPt, TokensEn

    def tf_encode(self, pt, en):
        """ doc """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
