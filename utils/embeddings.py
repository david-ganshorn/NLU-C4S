# ======================================================================================================================
# Copyright © 2017 David Ganshorn. All rights reserved.
# Created by David Ganshorn on 07/26/2017.
#
# Last Updated by David Ganshorn on 07/26/2017.
#
# ======================================================================================================================

import os

import numpy as np
import tensorflow as tf


def load(path, file, configuration):

    config = configuration

    f = tf.gfile.GFile(os.path.join(path, file), "r")

    model = {}

    for line in f:
        line = line.split()
        word = str(line[0])

        # config.word_dict.add(word)

        if config.word_dict.idx2str.__contains__(word):
            try:
                embedding = [float(val) for val in line[1:]]
                model[word] = np.array(embedding)
            except BaseException as e:
                """Do Nothing"""


    config.embeddings = model

    return model



def transpose(embeddings, configuration):

    config = configuration
    final_embeddings = {}

    # Transpose embedding keys to corresponding keys in dictionary
    # and order them in the correct order
    for word in config.word_dict.idx2str:
        embedding = embeddings.get(word)
        key = config.word_dict.get_index(word)
        final_embeddings[key] = embedding

    return final_embeddings


def get_vectors(embeddings, configuration):

    config = configuration

    vector = np.zeros([config.word_dict.size(), config.embedding_size])

    for word in embeddings:

        try:

            vector[word] = np.array(embeddings.get(word))
        except BaseException as e:
            print("Error:", word)

    return vector


def replace(questions, embeddings):

    unknown_words = []

    for item in questions:

        for word in item.tokens:

            if embeddings.keys().__contains__(word):
                ""
            else:
                unknown_words.append(word)
