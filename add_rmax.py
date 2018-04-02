#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:34:05 2018

Name: add_rmax.py

Purpose: add the property of rmax into the vortex structure

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv2'
__version__ = '1.00'  # consistent with the version of the C code
__date__ = '2018/02/26'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'

import numpy as np

def add_rmax(im_path, name='vortex.npz'):

    ds_dt = np.load(im_path + 'ds_dt.npz')
    nt = len(ds_dt['dt'])

    for i in range(nt):
        current = im_path + '{:d}'.format(i) + '/'
        vortex = dict(np.load(current + name))
        center = vortex['center']
        edge = vortex['edge']
        n = len(center)
        rmax = []
        for j in np.arange(n):
            e = edge[j]
            c = center[j]
            d = np.linalg.norm(np.subtract(e, c), axis=1)
            rmax.append(np.max(d))
        vortex['rmax'] = rmax
        np.savez(current + name, **vortex)


if __name__ == '__main__':
#    prefix = './SST/'
#    im_paths = [prefix + 'Swirl/6563core/',
#                prefix + 'Swirl/8542core/',
#                prefix + 'Swirl/6302wb/']
    prefix = './SOT/'
    im_paths = [prefix + 'Swirl/FG-blue/',
                prefix + 'Swirl/CaII-H/']
    for im_path in im_paths:
        add_rmax(im_path, name='vortex.npz')