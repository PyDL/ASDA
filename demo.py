#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday Apr 19 2019

Name: demo.py

Purpose: A demo of ASDA

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2019, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv3'
__date__ = '2019/04/19'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'


import numpy as np
from scipy.io import readsav
from datetime import datetime
from pyflct import flct, vcimageout
from vortex import gamma_values, center_edge, vortex_property

demo_file = 'demo_data.sav'
beg_time = datetime.today() # when you start to run this code
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
tau_file = 'tau_demo.dat'  # file in which Tau1 and Tau2 (see Liu et al. ApJ 2019) 
                      # will be stored
# Unit of the velocity will be pixel/frame
ds = 1.0
dt = 1.0
# Perform FLCT
vx, vy, vm = flct(data0, data1, dt, ds, 10,
                  outfile=vel_file)

# Perform swirl detection
factor = 1
# Tau1 and Tau2 
(tau1, tau2) = gamma_values(vx, vy, factor=factor)
# Store tau1 and tau2
vcimageout((tau1, tau2), tau_file)
# Determine Swirls
center, edge, points, peak, radius = center_edge(tau1, tau2,
                                                 factor=factor)
# Properties of Swirls
ve, vr, vc, ia = vortex_property(center, edge, points, vx, vy,
                                 data0)
# Save results
np.savez('vortex_demo.npz', center=center, edge=edge,
         points=points, peak=peak, radius=radius, ia=ia,
         ve=ve, vr=vr, vc=vc)

# Caculate time consumption
end_time = datetime.today()
print('Time used ', end_time-beg_time)
