#!/usr/bin/env python3
""" doc """


import numpy as np


def ngram(text, n):
    """ doc """
    unlist = 0
    if type(text[0]) is not list:
        text = [text]
        unlist = 1
    new_text = []
    for line in text:
        new_line = []
        for gram in range(len(line) - n + 1):
            new_gram = ""
            for i in range(n):
                if i != 0:
                    new_gram += " "
                new_gram += line[gram + i]
            new_line.append(new_gram)
        new_text.append(new_line)
    if unlist:
        return new_text[0]
    return new_text


def ngram_bleu(references, sentence, n):
    """ doc """
    references = ngram(references, n)
    sentence = ngram(sentence, n)
    dictt = {}
    for gram in sentence:
        dictt[gram] = dictt.get(gram, 0) + 1
    max_dict = {}
    for reference in references:
        this_ref = {}
        for gram in reference:
            this_ref[gram] = this_ref.get(gram, 0) + 1
        for gram in this_ref:
            max_dict[gram] = max(max_dict.get(gram, 0), this_ref[gram])
    in_ref = 0
    for gram in dictt:
        in_ref += min(max_dict.get(gram, 0), dictt[gram])
    closest = np.argmin(np.abs([len(ref) - len(sentence)
                        for ref in references]))
    closest = len(references[closest])
    if len(sentence) >= closest:
        brevity = 1
    else:
        brevity = np.exp(1 - (closest + n - 1) / (len(sentence) + n - 1))
    return brevity * in_ref / len(sentence)
