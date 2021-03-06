#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday Apr 19 2019

Name: demo.py

Purpose: A demo of ASDA

@author: Jaijia Liu at Queen's University Belfast

June 2021: update for version 2.2

Nov 2019: update for version 2
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2019, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv3'
__date__ = '2021/06/14'
__maintainor__ = 'Jiajia Liu'
__email__ = 'j.liu@qub.ac.uk'


import numpy as np
from scipy.io import readsav
from datetime import datetime
from asda.pyflct import flct, vcimageout, vcimagein
from asda.vortex import gamma_values, center_edge, vortex_property, read_vortex, save_vortex, gamma_values_parallel
import os
import subprocess
import multiprocessing as mp


if __name__ == '__main__':
    demo_file = 'demo_data.sav'
    # Read demo file
    demo = readsav(demo_file, python_dict=True)
    data0 = demo['data0']  # data in (y, x) order
    data1 = demo['data1']  # data in (y, x) order
    nx = data0.shape[1]
    ny = data0.shape[0]
    # Transpose data into (x, y) order
    data0 = data0.T
    data1 = data1.T

    vel_file = 'vel_demo.dat'  # file in which velocity field will be stored
    gamma_file = 'gamma_demo.dat'  # file in which Gamma1 and Gamma2 (see Liu et al. ApJ 2019)
                          # will be stored
    # Unit of the velocity will be pixel/frame
    ds = 1.0
    dt = 1.0
    # Perform FLCT
#    vx, vy, vm = flct(data0, data1, dt, ds, 10,
#                      outfile=vel_file)
    vx, vy, vm = vcimagein(vel_file)

    # Perform swirl detection
    factor = 1
    r = 3
    # Gamma1 and Gamma2
    beg_time = datetime.today() # when you start to run this code
    (gamma1, gamma2) = gamma_values(vx, vy, factor=factor)
    # a parallel version of gamma_values
    #(gamma1, gamma2) = gamma_values_parallel(vx, vy, r=r, factor=factor, ncpus=6)
    # Caculate time consumption
    end_time = datetime.today()
    print('Time used for calculating Gamma', end_time-beg_time)
    # Store gamma1 and gamma2
    vcimageout((gamma1, gamma2), gamma_file)
    # Determine Swirls
    center, edge, points, peak, radius = center_edge(gamma1, gamma2,
                                                     factor=factor)
    # Properties of Swirls
    ve, vr, vc, ia = vortex_property(center, edge, points, vx, vy,
                                     data0)
    # Save results
    vortex = {'center': center,
              'edge': edge,
              'points': points,
              'peak': peak,
              'radius': radius,
              've': ve,
              'vr': vr,
              'vc': vc,
              'ia': ia}
    save_vortex(vortex, filename='vortex_demo.npz')

    # perform comparison
    # compare between detection result and correct detection result
    # number of swirls
    correct = read_vortex(filename='correct.npz')
    n = len(ve)
    nc = len(correct['ve'])
    if n != nc:
        raise Exception("The number of swirls is wrong!")

    # find correspondances
    pos = []
    i = 0
    for cen in center:
        cen = [int(cen[0]), int(cen[1])]
        idx = np.where(correct['center'] == cen)
        if np.size(idx[0]) < 2:
            raise Exception("At least one swirl is not in the correct" +
                        " position")
        pos.append(np.bincount(idx[0]).argmax())

    # perform comparison
    peak_diff = []
    radius_diff = []
    vr_diff = []
    ve_diff = []
    vc_diff = []
    ia_diff = []
    for i in np.arange(n):
        idx = pos[i]
        peak_diff.append((peak[i] - correct['peak'][idx]) /
                            correct['peak'][idx])
        radius_diff.append((radius[i] -
                            correct['radius'][idx]) / correct['radius'][idx])
        vr_diff.append((vr[i] - correct['vr'][idx]) / correct['vr'][idx])
        ve_diff.append((ve[i] - correct['ve'][idx]) / correct['ve'][idx])
        vc_diff.append((vc[i] - correct['vc'][idx]) / correct['vc'][idx])
        ia_diff.append((ia[i] - correct['ia'][idx]) / correct['ia'][idx])

    print('Difference in Peak Gamma1 Value:', np.max(peak_diff),
            np.min(peak_diff))
    print('Difference in radius:', np.max(radius_diff),
            np.min(radius_diff))
    print('Difference in rotating speed:', np.max(vr_diff),
            np.min(vr_diff))
    print('Difference in expanding speed:', np.max(ve_diff),
            np.min(ve_diff))
    print('Difference in average intensity:', np.max(ia_diff),
            np.min(ia_diff))
