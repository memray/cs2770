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

    return config