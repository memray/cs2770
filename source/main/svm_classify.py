# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
import sklearn.preprocessing
import config_setting
from sklearn.metrics import confusion_matrix

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
    class_label = {}

    for class_id, (class_name, features) in enumerate(feature_dict.items()):
        data_list  = feature_dict[class_name]

        class_label[data_list[0]['class']] = class_name

        training_data['data'].extend(data_list[:int(len(data_list)*0.8)])
        validation_data['data'].extend(data_list[int(len(data_list)*0.8)+1: int(len(data_list)*0.9)])
        testing_data['data'].extend(data_list[int(len(data_list)*0.9)+1:])

    return training_data, validation_data, testing_data, class_label

def svm_test_model(feature_path):
    img_features = load_feature(feature_path)
    train_, validate_, test_, label_ = split_data(img_features)
    print(label_)

    X = sklearn.preprocessing.scale([d['feature'] for d in train_['data']])
    Y = [d['class'] for d in train_['data']]
    # X = X[500:1000]
    # Y = Y[500:1000]
    print(len(X))
    print(X.shape)
    # print(train_['data'][:50])

    lin_clf = svm.LinearSVC(max_iter=100, verbose=1, dual=False)
    lin_clf.fit(X, Y)

    X = sklearn.preprocessing.scale([d['feature'] for d in test_['data']])
    Y = [d['class'] for d in test_['data']]
    # X = X[500:1000]
    # Y = Y[500:1000]
    accuracy = lin_clf.score(X, Y)

    print('Accuracy(N=%d) = %f' % (len(X), accuracy))

    y_pred = lin_clf.predict(X)
    print(label_)
    cm = confusion_matrix(Y, y_pred)
    print(cm)




def transfer_training_plot():
    with open(config['local_model_dir']+ 'training_loss.pkl', 'rb') as f_:
        train_loss = pickle.load(f_)

    with open(config['local_model_dir']+ 'validate_accuracy.pkl', 'rb') as f_:
        validate_accuracy = pickle.load(f_)

    print(train_loss)
    print('-' * 50)
    print(validate_accuracy)

if __name__ == '__main__':
    # pretrain_feature = '../pretrain_feature_dump.pkl'
    newtrain_feature = '../newtrain_feature_dump.pkl'
    svm_test_model(newtrain_feature)
    #
    # transfer_training_plot()