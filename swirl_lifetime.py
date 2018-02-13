#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:16:38 2018

Name: swirl_lifetime.py

Purpose: Identify the same swirls in different frames and calculate their
         lifetime

My own notes:
    In the genrated label/info/lifetime files, there are three different
    methods:
        1. label/info/lifetime: compare two frames A and B, if a swirl in
           A has an estimated center with the region of a swirl in B, these
           two swirls will have the same label
        2. label3/info3/lifetime3: similar with method 1, but allow one frame
           missing (use three frames)
        3. label3_dmin/info3_dmin/lifetime3_dmin: similar with method 2, but
           use the distance between two swirls' centers to determine whether
           they will have the same label

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv2'
__version__ = '1.00'  # consistent with the version of the C code
__date__ = '2018/01/03'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'

import numpy as np
from matplotlib.path import Path


def get_label(vortex, pvortex, plabel, p2vortex=[], p2label=[], dmin=False,
              three_frame=False):
    '''
    Input:
        vortex: vortex information of current frame
        pvortex: vortex information of previous frame
        plabel: label information of previous frame
        p2vortex, p2label: vortex and label of 2 frames before
        dmin: if true, will use the distance between centers of swirls to
              determine whether they will have the same label.
        three_frame: compare with 2 frames before
    Output:
        label: label information of current frame
    '''
    n_vortex = len(vortex['radius'])
    n_pvortex = len(plabel)
    step = 1
    npre = n_pvortex
    n_p2vortex = len(p2label)
    if three_frame:
        step = 2
        npre = n_p2vortex
    label = np.zeros(n_vortex, dtype=int)
    if n_vortex != 0:
        for i in range(n_vortex):
            edge = vortex['edge'][i]
            center = vortex['center'][i]
            path_edge = Path(edge)
            index = []
            dist = []
            selected = []
            for j in range(npre):
                if not three_frame:
                    pcenter = pvortex['center'][j]
                    pvc = pvortex['vc'][j]
                else:
                    pcenter = p2vortex['center'][j]
                    pvc = p2vortex['vc'][j]
                exp_center = [pcenter[0]+pvc[0]*step, pcenter[1]+pvc[1]*step]
                contain = path_edge.contains_point(exp_center)
                d = np.linalg.norm([center[0]-exp_center[0],
                                    center[1]-exp_center[1]])
                if dmin:
                    if d <= vortex['radius'][i]:
                        contain = True
                if contain:
                    index.append(j)
                    if not three_frame:
                        selected.append(plabel[j])
                    else:
                        selected.append(p2label[j])
                    dist.append(d)
            if index == []:
                if n_pvortex == 0:
                    label[i] = np.max([np.max(label), np.max(p2label)]) + 1
                else:
                    label[i] = np.max([np.max(label), np.max(plabel)]) + 1
            else:
                idx = np.where(dist == np.min(dist))
                idx = index[idx[0][0]]
                label[i] = plabel[idx] if not three_frame else p2label[idx]
    return label


def group_swirls(im_path, label_name='label', dmin=False, three_frame=False):
    '''
    Give detected swirls in different frames group numbers
    If two/three... swirls in two/three... frames have the same group number,
    they will be considered as the same swirl
    '''
    ds_dt = np.load(im_path + 'ds_dt.npz')
    nt = len(ds_dt['dt'])
    vortex = dict(np.load(im_path + '0/vortex.npz'))
    n_vortex = len(vortex['radius'])
    label = np.arange(0, n_vortex)
    np.save(im_path + '0/' + label_name+'.npy', label)
    for i in np.arange(1, nt):
        vortex_file = im_path + "{:d}".format(i) + '/vortex.npz'
        pvortex_file = im_path + "{:d}".format(i-1) + '/vortex.npz'
        label_file = im_path + "{:d}".format(i) + '/' + label_name+'.npy'
        plabel_file = im_path + "{:d}".format(i-1) + '/' + label_name+'.npy'
        vortex = dict(np.load(vortex_file))
        pvortex = dict(np.load(pvortex_file))
        plabel = np.load(plabel_file)
        p2vortex = []
        p2label = []
        if i == 1:
            label = get_label(vortex, pvortex, plabel, dmin=dmin,
                              three_frame=False)
        else:
            p2vortex_file = im_path + "{:d}".format(i-2) + '/vortex.npz'
            p2label_file = im_path + "{:d}".format(i-2) + '/' + \
                            label_name+'.npy'
            p2vortex = dict(np.load(p2vortex_file))
            p2label = np.load(p2label_file)
            label = get_label(vortex, pvortex, plabel, p2vortex,
                              p2label, dmin=dmin, three_frame=False)
            if three_frame:
                label2 = get_label(vortex, pvortex, plabel, p2vortex,
                                   p2label, dmin=dmin, three_frame=True)
                label = np.minimum(label, label2)

        np.save(label_file, label)


def label_frames(im_path, label_name='label', info_name='label_info'):
    '''
    If there are n different swirls (the last number of label is then n),
    generate a file containing a numpy array, which stores which frames
    a label appears in.
    '''
    ds_dt = np.load(im_path + 'ds_dt.npz')
    nt = len(ds_dt['dt'])
    label = np.load(im_path + '{:d}'.format(nt-1) + '/' + label_name+'.npy')
    n_label = np.max(label) + 1
    label_info = {}
    labels = {}
    for i in range(nt):
        label_file = im_path + "{:d}".format(i) + '/' + label_name+'.npy'
        label = np.load(label_file)
        labels[i] = label
    for j in range(n_label):
        key = "{:d}".format(j)
        for i in range(nt):
            if j in labels[i]:
                if key not in label_info.keys():
                    label_info[key] = [i]
                else:
                    label_info[key].append(i)
    info_file = im_path + info_name + '.npz'
    np.savez(info_file, **label_info)
    return label_info


if __name__ == '__main__':
    prefix = './'
    im_paths = [prefix + 'SST/Swirl/6302wb/',
                prefix + 'SST/Swirl/6563core/',
                prefix + 'SST/Swirl/8542core/']
    label_name = 'label'
    info_name = 'label_info'
    three_frame = False
    if three_frame:
        label_name = label_name + '3'
        info_name = info_name + '3'
    dmin = False
    if dmin:
        label_name = label_name + '_dmin'
        info_name = info_name + '_dmin'
    for im_path in im_paths:
        group_swirls(im_path, label_name=label_name, dmin=dmin,
                     three_frame=three_frame)
        label_info = label_frames(im_path, label_name=label_name,
                                  info_name=info_name)
