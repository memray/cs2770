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

def training():
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

    training_data = {}
    training_data['data'] = []
    training_data['label'] = []
    validation_data = {}
    validation_data['data'] = []
    validation_data['label'] = []
    testing_data = {}
    testing_data['data'] = []
    testing_data['label'] = []

    if os.path.exists(config['training_data_cache']):
        print('Found processed data, loading...')

        with open(config['training_data_cache'], 'rb') as f_:
            training_data = pickle.load(f_)
        with open(config['validation_data_cache'], 'rb') as f_:
            validation_data = pickle.load(f_)
        with open(config['testing_data_cache'], 'rb') as f_:
            testing_data = pickle.load(f_)

    else:
        for class_id, class_name in enumerate(os.listdir(config['data_path'])):
            print("Processing - " + class_name)
            data_list = []
            for img_id, img_file in enumerate(os.listdir(config['data_path']+os.sep+class_name)):
                img = caffe.io.load_image(config['data_path']+os.sep+class_name+os.sep+img_file)
                img = transformer.preprocess('data', img)
                data_list.append(img)
            label_list = [class_id] * len(data_list)
            training_data['data'].extend(data_list[:int(len(data_list)*0.8)])
            training_data['label'].extend(label_list[:int(len(data_list)*0.8)])
            validation_data['data'].extend(data_list[int(len(data_list)*0.8)+1: int(len(data_list)*0.9)])
            validation_data['label'].extend(label_list[int(len(data_list)*0.8)+1:int(len(data_list)*0.9)])
            testing_data['data'].extend(data_list[int(len(data_list)*0.9)+1:])
            testing_data['label'].extend(label_list[int(len(data_list)*0.9)+1:])

        print('#(Training)=%d' % len(training_data['data']))
        print('#(Validation)=%d' % len(validation_data['data']))
        print('#(Testing)=%d' % len(testing_data['data']))

        with open(config['training_data_cache'], 'wb') as f_:
            pickle.dump(training_data, f_, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config['validation_data_cache'], 'wb') as f_:
            pickle.dump(validation_data, f_, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config['testing_data_cache'], 'wb') as f_:
            pickle.dump(testing_data, f_, protocol=pickle.HIGHEST_PROTOCOL)

    training_data['data'] = np.asarray(training_data['data'])
    training_data['label'] = np.asarray(training_data['label'])
    train_loss = []
    validate_accuracy = []

    for epoch in range(25):
        '''
        Training
        '''
        print('Training epoch=%d' % epoch)
        number_data =  len(training_data['data'])
        number_minibatch = number_data/config['minibatch_size']+1
        shuffled_index = np.arange(number_data)
        np.random.shuffle(shuffled_index)
        print('\t #(minibatch)=%d' % number_minibatch)

        for it in range(number_minibatch):
            print('\tTraining epoch=%d, round=%d' % (epoch, it))

            if it == number_minibatch-1:
                data_ = training_data['data'][shuffled_index[it * config['minibatch_size']: number_data]]
                label_ = training_data['label'][shuffled_index[it * config['minibatch_size']: number_data]]
                print('Training %d-%d, size(data_)=%d, size(label_)=%d' % (it * config['minibatch_size'], number_data, len(data_), len(label_)))

            else:
                data_ = training_data['data'][shuffled_index[it * config['minibatch_size']: (it + 1) * config['minibatch_size']]]
                label_ = training_data['label'][shuffled_index[it * config['minibatch_size']: (it + 1) * config['minibatch_size']]]
                print('Training %d-%d, size(data_)=%d, size(label_)=%d' % (it * config['minibatch_size'], (it + 1) * config['minibatch_size'], len(data_), len(label_)))

            if len(data_) != config['minibatch_size']:
                break

            solver.net.blobs['data'].data[...] = data_
            solver.net.blobs['label'].data[...] = label_

            solver.step(1)

            loss = solver.net.blobs['loss'].data
            train_loss.append(loss)
            print('iteration %d, loss = %f' % (len(train_loss), loss))

        '''
        Validation
        '''
        print('Validating epoch=%d' % epoch)

        validate_a = []
        number_data = len(validation_data['data'])
        number_minibatch = number_data/config['minibatch_size']+1
        for it in range(number_minibatch):

            if it == number_minibatch-1:
                data_ = validation_data['data'][it * config['minibatch_size']: number_data]
                label_ = validation_data['label'][it * config['minibatch_size']: number_data]
                print('Validating %d-%d, size(data_)=%d, size(label_)=%d' % (it * config['minibatch_size'], number_data, len(data_), len(label_)))
            else:
                data_ = validation_data['data'][it * config['minibatch_size']: (it + 1) * config['minibatch_size']]
                label_ = validation_data['label'][it * config['minibatch_size']: (it + 1) * config['minibatch_size']]
                print('Validating %d-%d, size(data_)=%d, size(label_)=%d' % (it * config['minibatch_size'], (it + 1) * config['minibatch_size'], len(data_), len(label_)))

            if len(data_) != config['minibatch_size']:
                break

            solver.net.blobs['data'].data[...] = data_
            solver.net.blobs['label'].data[...] = label_

            solver.net.forward()
            validate_a.append(solver.net.blobs['accuracy'].data)

        validate_accuracy.append(np.average(validate_a))
        print('Epoch %d, accuracy = %f' % (epoch, np.average(validate_a)))

        print('-' * 50)

    solver.net.save(config['trained_model_dir']+'model.epoch=%d.caffemodel' % epoch)
    with open(config['trained_model_dir']+ 'training_loss.pkl', 'w') as f_:
        pickle.dump(train_loss, f_, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['trained_model_dir']+ 'validate_accuracy.pkl', 'w') as f_:
        pickle.dump(validate_accuracy, f_, protocol=pickle.HIGHEST_PROTOCOL)

def testing():
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
    solver.net.copy_from(config['trained_model_dir']+'model.epoch=24.caffemodel')

    testing_data = {}
    testing_data['data'] = []
    testing_data['label'] = []

    if not os.path.exists(config['training_data_cache']):
        print('Not found processed data')
        exit()

    with open(config['testing_data_cache'], 'rb') as f_:
        testing_data = pickle.load(f_)

    test_a = []
    number_data = len(testing_data['data'])
    number_minibatch = number_data / config['minibatch_size'] + 1
    for it in range(number_minibatch):

        if it == number_minibatch - 1:
            data_ = testing_data['data'][it * config['minibatch_size']: number_data]
            label_ = testing_data['label'][it * config['minibatch_size']: number_data]
            print('Validating %d-%d, size(data_)=%d, size(label_)=%d' % (
            it * config['minibatch_size'], number_data, len(data_), len(label_)))
        else:
            data_ = testing_data['data'][it * config['minibatch_size']: (it + 1) * config['minibatch_size']]
            label_ = testing_data['label'][it * config['minibatch_size']: (it + 1) * config['minibatch_size']]
            print('Validating %d-%d, size(data_)=%d, size(label_)=%d' % (
            it * config['minibatch_size'], (it + 1) * config['minibatch_size'], len(data_), len(label_)))

        if len(data_) != config['minibatch_size']:
            break

        solver.net.blobs['data'].data[...] = data_
        solver.net.blobs['label'].data[...] = label_

        solver.net.forward()
        test_a.append(solver.net.blobs['accuracy'].data)

    print('Test accuracy = %f' % (np.average(test_a)))

def extract_feature():
    caffe.set_device(3) # ENTER THE GPU NUMBER YOU NOTED ABOVE (0-3) HERE
    caffe.set_mode_gpu()

    config = config_setting.load_config()
    net = caffe.Net('/tmp/caffe/models/deploy.prototxt', config['trained_model_dir']+'model.epoch=24.caffemodel', caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    solver = caffe.SGDSolver(config['solver.prototxt'])
    solver.net.copy_from(config['trained_model_dir']+'model.epoch=24.caffemodel')

    testing_data = {}
    testing_data['data'] = []
    testing_data['label'] = []

    if not os.path.exists(config['training_data_cache']):
        print('Not found processed data')
        exit()

    with open(config['testing_data_cache'], 'rb') as f_:
        testing_data = pickle.load(f_)

    test_a = []
    number_data = len(testing_data['data'])
    number_minibatch = number_data / config['minibatch_size'] + 1
    for it in range(number_minibatch):

        if it == number_minibatch - 1:
            data_ = testing_data['data'][it * config['minibatch_size']: number_data]
            label_ = testing_data['label'][it * config['minibatch_size']: number_data]
            print('Validating %d-%d, size(data_)=%d, size(label_)=%d' % (
            it * config['minibatch_size'], number_data, len(data_), len(label_)))
        else:
            data_ = testing_data['data'][it * config['minibatch_size']: (it + 1) * config['minibatch_size']]
            label_ = testing_data['label'][it * config['minibatch_size']: (it + 1) * config['minibatch_size']]
            print('Validating %d-%d, size(data_)=%d, size(label_)=%d' % (
            it * config['minibatch_size'], (it + 1) * config['minibatch_size'], len(data_), len(label_)))

        if len(data_) != config['minibatch_size']:
            break

        solver.net.blobs['data'].data[...] = data_
        solver.net.blobs['label'].data[...] = label_

        solver.net.forward()
        test_a.append(solver.net.blobs['accuracy'].data)

    print('Test accuracy = %f' % (np.average(test_a)))


if __name__ == '__main__':
    testing()