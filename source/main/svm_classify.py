# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
import sklearn.preprocessing
import config_setting

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

config = config_setting.load_config()

def load_feature(path):
    with open(path, 'rb') as f_: #config['feature_path']
        img_features = pickle.load(f_)

    return img_features


def split_data(feature_dict):
    training_data = {}
    training_data['data'] = []
    training_data['label'] = []
    validation_data = {}
    validation_data['data'] = []
    validation_data['label'] = []
    testing_data = {}
    testing_data['data'] = []
    testing_data['label'] = []

    for class_id, (class_name, features) in enumerate(feature_dict.items()):
        data_list  = feature_dict[class_name]

        training_data['data'].extend(data_list[:int(len(data_list)*0.8)])
        validation_data['data'].extend(data_list[int(len(data_list)*0.8)+1: int(len(data_list)*0.9)])
        testing_data['data'].extend(data_list[int(len(data_list)*0.9)+1:])


    return training_data, validation_data, testing_data

def svm_test_model(feature_path):
    img_features = load_feature(feature_path)
    train_, validate_, test_ = split_data(img_features)

    X = sklearn.preprocessing.scale([d['feature_fc8'] for d in train_['data']])
    Y = [d['class'] for d in train_['data']]
    print(len(X))
    print(train_['data'][:5])

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X, Y)

    X = sklearn.preprocessing.scale([d['feature_fc8'] for d in test_['data']])
    Y = [d['class'] for d in test_['data']]
    accuracy = lin_clf.score(X, Y)

    print('Accuracy on %d data = %f' % (len(X), accuracy))

if __name__ == '__main__':
    pretrain_feature = '../pretrain_feature_dump.pkl'
    newtrain_feature = '../newtrained_feature_dump.pkl'
    svm_test_model(newtrain_feature)