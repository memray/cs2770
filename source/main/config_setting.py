#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def load_config():
    config = {}
    config['data_path']  = '/tmp/caffe/data/' #'/Users/memray/Project/Course/cs2770/data/'
    config['proto_path'] = '/tmp/caffe/models/deploy.prototxt'
    config['caffemodel_path'] = '/tmp/caffe/models/weights.caffemodel'
    config['feature_path'] = '/afs/cs.pitt.edu/usr0/memray/private/feature_dump.pkl'

    config['train_val.prototxt'] = '/afs/cs.pitt.edu/usr0/memray/private/cs2770/models/train_val.prototxt'
    config['solver.prototxt'] = '/afs/cs.pitt.edu/usr0/memray/private/cs2770/models/solver.prototxt'
    config['deploy.prototxt'] = '/afs/cs.pitt.edu/usr0/memray/private/cs2770/models/deploy.prototxt'
    config['minibatch_size'] = 1
    config['trained_model_dir'] = '/afs/cs.pitt.edu/usr0/memray/private/trained_models.stepsize=1000.lr=0.001/'

    if not os.path.exists(config['trained_model_dir']):
        os.makedirs(config['trained_model_dir'])

    config['training_data_cache'] = '/afs/cs.pitt.edu/usr0/memray/private/processed_data/training_data.pkl'
    config['validation_data_cache'] = '/afs/cs.pitt.edu/usr0/memray/private/processed_data/validation_data.pkl'
    config['testing_data_cache'] = '/afs/cs.pitt.edu/usr0/memray/private/processed_data/testing_data.pkl'

    return config