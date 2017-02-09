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
    config['epoch_size'] = 32
    config['trained_model_dir'] = '/afs/cs.pitt.edu/usr0/memray/private/trained_models/'

    return config