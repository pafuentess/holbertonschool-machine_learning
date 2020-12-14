#!/usr/bin/env python3
""" doc """

import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def preprocessing():
    """ doc """
    filename = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    data = pd.read_csv(filename)
    # borra lo que estaba en cero
    df = data.dropna()
    # Convert argument to datetime.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df[df['Timestamp'].dt.year >= 2017]
    # Reset the index, or a level of it.
    df.reset_index(inplace=True, drop=True)
    # print(df)

    df = df[0::60]
    datetime = pd.to_datetime(df.pop('Timestamp'))
    variables = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)',
                 'Volume_(Currency)', 'Weighted_Price']
    # print(df.describe().transpose)

    plot_features = df[variables]
    plot_features.index = datetime
    _ = plot_features.plot(subplots=True)

    # Split the data
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    plot_features = train_df[variables]
    plot_features.index = datetime[0:int(n * 0.7)]
    _ = plot_features.plot(subplots=True)
    # plt.show()

    # Normalize the data
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
