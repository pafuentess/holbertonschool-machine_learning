#!/usr/bin/env python3
""" doc """

import numpy as np


def uni_bleu(references, sentence):
    """ doc """
    uniques = list(set(sentence))
    dict_words = {}
    len_cand = len(sentence)
    best = []

    for reference in references:
        for word in reference:
            if word in uniques:
                if word not in dict_words.keys():
                    dict_words[word] = reference.count(word)
                else:
                    actual = reference.count(word)
                    prev = dict_words[word]
                    dict_words[word] = max(actual, prev)

    prob = sum(dict_words.values()) / len_cand

    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - len_cand)
        best.append((diff, ref_len))

    sort = sorted(best, key=lambda x: x[0])
    best_match = sort[0][1]

    if len_cand > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (best_match / len_cand))

    Bleu = bp * np.exp(np.log(prob))
    if Bleu > 0.4:
        return round(Bleu, 7)
    return Bleu
