#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """ doc """
    return model.wv.get_keras_embedding(train_embeddings=True)
