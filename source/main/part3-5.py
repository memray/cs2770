# -*- coding: utf-8 -*-

import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
import caffe
import config_setting

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    caffe.set_device(3) # ENTER THE GPU NUMBER YOU NOTED ABOVE (0-3) HERE
    caffe.set_mode_gpu()

    config = config_setting.load_config()
    net = caffe.Net('/tmp/caffe/models/deploy.prototxt', '/tmp/caffe/models/weights.caffemodel', caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    solver = caffe.SGDSolver(config['solver.prototxt'])
    solver.net.copy_from('/tmp/caffe/models/weights.caffemodel')

    training_data = []
    validation_data = []
    testing_data = []

    for class_id, class_name in enumerate(os.listdir(config['data_path'])):
        print(class_name)
        data_list = []
        for img_id, img_file in enumerate(os.listdir(config['data_path']+os.sep+class_name)):
            img = caffe.io.load_image(config['data_path']+os.sep+class_name+os.sep+img_file)
            img = transformer.preprocess('data', img)
            data_list.append(img)
        label_list = [class_id] * len(data_list)
        training_data.extend([data_list[:len(data_list)*0.8], label_list[:len(data_list)*0.8]])
        validation_data.extend([data_list[len(data_list)*0.8+1:len(data_list)*0.9], label_list[len(data_list)*0.8+1:len(data_list)*0.9]])
        testing_data.extend([data_list[len(data_list)*0.9+1:], label_list[len(data_list)*0.9+1:]])

    training_data = np.asarray(training_data)
    train_loss = []
    validate_accuracy = []

    for epoch in range(1):
        '''
        Training
        '''
        print('Training epoch=%d' % epoch)
        shuffled_index = np.arange(len(training_data))
        np.random.shuffle(shuffled_index)

        for it in range(len(training_data)/config['epoch_size']+1):

            if it == len(training_data)/config['epoch_size']:
                data_ = training_data[shuffled_index[it * config['epoch_size']: len(training_data) - 1]]
                print('Training %d-%d' % (it * config['epoch_size'], len(training_data) - 1))
            else:
                data_ = training_data[shuffled_index[it * config['epoch_size']: (it + 1) * config['epoch_size'] - 1]]
                print('Training %d-%d' % (it * config['epoch_size'], (it + 1) * config['epoch_size'] - 1))

            solver.net.blobs['data'].data[...] = data_[0]
            solver.net.blobs['label'].data[...] = data_[1]

            solver.step(1)
            train_loss.append(solver.net.blobs['loss'].data)

        '''
        Validation
        '''
        print('Validating epoch=%d' % epoch)

        validate_a = []
        for it in range(len(validation_data)/config['epoch_size']+1):

            if it == len(validation_data)/config['epoch_size']:
                data_ = validation_data[it * config['epoch_size']: len(validation_data) - 1]
            else:
                data_ = validation_data[it * config['epoch_size']: (it + 1) * config['epoch_size'] - 1]

            solver.net.blobs['data'].data[...] = data_[0]
            solver.net.blobs['label'].data[...] = data_[1]

            solver.net.forward()
            validate_a.append(solver.net.blobs['accuracy'].data)

        validate_accuracy.append(np.average(validate_a))

        solver.net.save(config['trained_model_dir']+'model.epoch=%d.caffemodel' % epoch)

    with open(config['trained_model_dir']+ 'training_loss.pkl', 'w') as f_:
        pickle.dump(train_loss, f_, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['trained_model_dir']+ 'validate_accuracy.pkl', 'w') as f_:
        pickle.dump(validate_accuracy, f_, protocol=pickle.HIGHEST_PROTOCOL)