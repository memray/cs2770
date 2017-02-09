# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
import sklearn.preprocessing
import caffe
import config_setting

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

config = config_setting.load_config()

def extract_feature(deploy_proto,  model_path):
    caffe.set_device(3) # ENTER THE GPU NUMBER YOU NOTED ABOVE (0-3) HERE
    caffe.set_mode_gpu()


    net = caffe.Net(deploy_proto, model_path, caffe.TEST)
    # net = caffe.Net(config['proto_path'], config['caffemodel_path'], caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # data_dir = '/tmp/caffe/data/sofa'
    # img_file = data_dir + '/2011_001451.jpg'

    img_features = {}
    for class_id, class_name in enumerate(os.listdir(config['data_path'])):
        print(class_name)
        img_features[class_name] = [];

        for img_id, img_file in enumerate(os.listdir(config['data_path']+os.sep+class_name)):
            if img_id % 100 == 0:
                print(img_id)
                print(config['data_path']+os.sep+class_name+os.sep+img_file)

            data_sample = {}
            data_sample['name'] = img_file
            data_sample['class'] = class_id
            data_sample['class_name'] = class_name

            print('Image Matrix:')
            img = caffe.io.load_image(config['data_path']+os.sep+class_name+os.sep+img_file)
            img = transformer.preprocess('data', img)

            print(img)

            net.blobs['data'].data[...] = img
            output = net.forward()

            data_sample['feature'] = copy.deepcopy(net.blobs['fc7'].data[0])

            print('Feature Matrix:')
            print(net.blobs['fc7'].data[0])
            input("Press enter!")
            print('*' * 50)
            # data_sample['feature_fc8'] = net.blobs['fc8_class20'].data[0]
            # 'fc8' layer in pretrain AlexNet
            # data_sample['feature_fc8'] = net.blobs['fc8'].data[0]

            img_features[class_name].append(data_sample)

    with open(config['feature_path'], 'wb') as handle:
        pickle.dump(img_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # extract_feature('/tmp/caffe/models/weights.caffemodel')
    extract_feature(config['deploy.prototxt'], config['trained_model_dir']+'model.epoch=24.caffemodel')