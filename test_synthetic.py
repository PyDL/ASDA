#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:22:57 2018

Name: test_synthetic.py

Purpose: Test synthetic vortices

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv2'
__version__ = '1.00'  # consistent with the version of the C code
__date__ = '2018/01/18'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'


import numpy as np
from lamb_oseen import Lamb_Oseen
from pyflct import vcimagein, vcimageout
from vortex import center_edge, vortex_property
import random
import os
import matplotlib.pyplot as plt
import subprocess
import multiprocessing


def generate_synthetic(noise_level=0.1,
                       outfile='./SST/Synthetic/noise_0.1.dat',
                       infofile='./SST/Synthetic/info_0.1.npz',
                       nx=5000, ny=5000, n=1000):

    # nx, ny: size of the image
    # n: number of swirls

    xran = range(nx)
    yran = range(ny)
    xx, yy = np.meshgrid(xran, yran)

    # radius with gaussian distribution
    rmiu = 7.2
    rsig = 1.6
    radius = np.random.normal(rmiu, rsig, n)
    # rotating speed with gaussian distribution
    # for photospheric swirls
    vmiu = 0.17
    vsig = 0.07
    # for chromospheric swirls, vmiu=0.7, vsig=0.3
    vtheta0 = np.random.normal(vmiu, vsig, int(n/2))
    vtheta1 = np.random.normal(0-vmiu, vsig, n - int(n/2))
    vtheta = np.append(vtheta0, vtheta1)

    # velocity with random background noise
    vnoise = vmiu / np.sqrt(2.0) * noise_level
    vx = (np.random.rand(nx, ny) - 0.5) * vnoise * 2.0
    vy = (np.random.rand(nx, ny) - 0.5) * vnoise * 2.0
    vx = vx.T
    vy = vy.T
    mask = np.ones((nx, ny))
    mask = mask.T

    loc = []
    # Generate Lamb-Oseen Vortices and insert them into vx and vy
    for i in range(n):
        vmax = vtheta[i]
        rmax = radius[i]
        arange = [int(-1.5 * rmax), int(1.5 * rmax)]
        lo = Lamb_Oseen(vmax, rmax)
        x, y = lo.get_grid(xrange=arange, yrange=arange)
        vxi, vyi = lo.get_vxvy(x, y)
        vxi = vxi.T
        vyi = vyi.T
        flag = True
        while flag:
            pi = random.choice(xran)
            pj = random.choice(yran)
            if mask[pi, pj] == 1 and pi-arange[1] > 0 and pi+arange[1] < nx-1 \
               and pj-arange[1] > 0 and pj+arange[1] < ny-1:
                flag = False
                for ii in np.arange(pi-arange[1], pi+arange[1]):
                    for jj in np.arange(pj-arange[1], pj+arange[1]):
                        if mask[ii, jj] == 0:
                            flag = True
        loc.append([pi, pj])
        vx[pi-arange[1]:pi+arange[1], pj-arange[1]:pj+arange[1]] = \
            np.add(vx[pi-arange[1]:pi+arange[1], pj-arange[1]:pj+arange[1]],
                   vxi)
        vy[pi-arange[1]:pi+arange[1], pj-arange[1]:pj+arange[1]] = \
            np.add(vy[pi-arange[1]:pi+arange[1], pj-arange[1]:pj+arange[1]],
                   vyi)
        mask[pi-arange[1]:pi+arange[1], pj-arange[1]:pj+arange[1]] = 0

    vcimageout([vx, vy, mask], outfile)
    np.savez(infofile, radius=radius, vtheta=vtheta, loc=loc)
    return vx, vy, mask, radius, vtheta, loc


def get_detection(gamma1, gamma2, rmin=1.0, loc=None, rad=None):
    '''
    Detect swirls from gamma1 and gamma2
    if loc is not None, detected swirls will be sorted in the same order of
    the input swirls
    '''
    center, edge, points, peak, radius = center_edge(gamma1, gamma2,
                                                     rmin=rmin)
    ve, vr, vc, ia = vortex_property(center, edge, points, vx, vy)
    vortex = {'center': center, 'edge': edge, 'points': points, 'peak': peak,
              'radius': radius, 've': ve, 'vr':vr, 'ia': ia}
    n = len(radius)
    # Flag for false detection, 1 for false detection
    false = np.zeros(n, dtype=int)
    index = []
    if loc is not None and rad is not None:
        norig = len(rad)
        for i in range(n):
            cen = center[i]
            dis = 1e20
            idx = 0
            for j in range(norig):
                if np.linalg.norm(np.subtract(cen, loc[j])) < dis:
                    idx = j
                    dis = np.linalg.norm(np.subtract(cen, loc[j]))
            index.append(idx)
            if np.linalg.norm(np.subtract(cen, loc[idx])) >= \
               rad[idx] + radius[i]:
                false[i] = 1
        vortex['index'] = index
        vortex['false'] = false

    return vortex


def compare_figures(vortex, info, vx, vy, file):
    '''
    Generate figures comparing the detection result and the original input
    vortex
    '''
    plt.figure(figsize=(10, 10))

    low = 2400
    high = 2600
    # Panel (a)
    plt.subplot(2, 2, 1)
    x, y = zip(*info['loc'])
    xd, yd = zip(*vortex['center'])
    plt.scatter(x, y, c='green', s=25)
    plt.scatter(xd, yd, c='orange', s=2)
    plt.title('(a) Location of Vortices')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.plot([low, high, high, low, low], [low, low, high, high, low],
             c='black')

    # Panel (b)
    xx, yy = np.meshgrid(np.arange(low, high), np.arange(low, high))
    plt.subplot(2, 2, 2)
    vx0 = vx[low:high, low:high]
    vy0 = vy[low:high, low:high]
    st = 3
    plt.quiver(xx[::st, ::st], yy[::st, ::st],
               vx0.T[::st, ::st], vy0.T[::st, ::st],
               color='green', linewidth=0.5)
    for i in range(len(xd)):
        if xd[i] >= low and xd[i] <= high and yd[i] >= low and \
           yd[i] <= high:
            c = 'blue' if vortex['vr'][i] > 0 else 'red'
            xe, ye = zip(*vortex['edge'][i])
            plt.plot(xe, ye, c=c, linewidth=1)
            plt.scatter(xd[i], yd[i], c=c, s=10)
    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('(b) Velocity Field in the Black Box')

    # Panel (c)
    plt.subplot(2, 2, 3)
    plt.hist(info['radius'], bins=np.arange(0, 14, 0.5), histtype='step',
             color='green')
    plt.hist(vortex['radius'], bins=np.arange(0, 14, 0.5), histtype='step',
             color='orange')
    plt.xlim(0, 15)
    plt.xlabel('Radius')
    plt.ylabel('Number of Vortices')
    plt.title('(c) Distibution of Radius')

    # Panel (d)
    plt.subplot(2, 2, 4)
    plt.hist(info['vtheta'], bins=np.arange(-0.5, 0.5, 0.05), histtype='step',
             color='green')
    plt.hist(vortex['vr'], bins=np.arange(-0.5, 0.5, 0.05), histtype='step',
             color='orange')
    plt.xlim(-0.5, 0.5)
    plt.xlabel('Rotating Speed')
    plt.ylabel('Number of Vortices')
    plt.title('(d) Distibution of Rotating Speed')

    plt.show()
    plt.savefig(file, dpi=300)
    plt.close()


def detection_rate(vortex, info, noise_level):
    '''
    Print out the detection rate, false detection rate, location/radius/
    velocity accuracies.
    '''
    print('Noise Level:', noise_level)
    x, y = zip(*info['loc'])
    xd, yd = zip(*vortex['center'])
    print('Detection Rate:', len(xd) * 1.0 / len(x) * 100, '%')
    false = vortex['false']
    idx = np.where(false == 1)[0]
    print('False Detection:', len(idx) * 1.0 / len(false) * 100, '%')

    n = len(xd)
    full = np.zeros(len(x))
    loc_acc = []
    rad_acc = []
    vr_acc = []
    for i in range(n):
        idx = vortex['index'][i]
        full[idx] = 1
        cend = vortex['center'][i]
        radd = vortex['radius'][i]
        vrd = vortex['vr'][i]
        cen = info['loc'][idx]
        rad = info['radius'][idx]
        vr = info['vtheta'][idx]

        rad_acc.append(np.abs(rad - radd) / rad)
        vr_acc.append(np.abs(vrd - vr) / np.abs(vr))
        loc_acc.append(np.linalg.norm(np.subtract(cend, cen)) / rad)
    print('Average Location Accuracy', (1-np.mean(loc_acc))*100, '%')
    print('Worst Location Accuracy', (1-np.max(loc_acc))*100, '%')
    print('Average Radius Accuracy', (1-np.mean(rad_acc))*100, '%')
    print('Worst Radius Accuracy', (1-np.max(rad_acc))*100, '%')
    print('Average Rotating Speed Accuracy', (1-np.mean(vr_acc))*100, '%')
    print('Worst Rotating Speed Accuracy', (1-np.max(vr_acc))*100, '%')


if __name__ == '__main__':

    ncpu = multiprocessing.cpu_count()
    root_path = './SST/Synthetic/'
    noise_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    for noise_level in noise_levels:
        velfile = root_path + 'noise_' + str(noise_level) + '.dat'
        infofile = root_path + 'info_' + str(noise_level) + '.npz'
        gammafile = root_path + 'gamma_' + str(noise_level) + '.dat'
        vortexfile = root_path + 'vortex_' + str(noise_level) + '.npz'
        if os.path.exists(velfile) and os.path.exists(infofile):
            vx, vy, mask = vcimagein(velfile)
            info = np.load(infofile)
            rad = info['radius']
            vtheta = info['vtheta']
            loc = info['loc']
        else:
            vx, vy, mask, rad, vtheta, loc = \
                generate_synthetic(noise_level, velfile, infofile)
            info = np.load(infofile)
        if os.path.exists(gammafile):
            gamma1, gamma2 = vcimagein(gammafile)
        else:
            factor = 1.0
            code = '#!/bin/bash \n' + \
                   'cd ' + os.getcwd() + ' \n' + \
                   'source activate root \n' + \
                   'mpirun -np ' + str(ncpu) + ' python' + \
                   ' gamma_values_mpi.py' + \
                   ' -f ' + velfile + \
                   ' -r 3' + \
                   ' -a '+str(factor) + \
                   ' -o ' + gammafile
            f = open('./tmp.sh', 'w')
            f.write(code)
            f.close()
            process = subprocess.Popen(['bash', './tmp.sh'])
            process.wait()
            os.remove('./tmp.sh')
            (gamma1, gamma2) = vcimagein(gammafile)
        if not os.path.exists(vortexfile):
            vortex = get_detection(gamma1, gamma2, loc=loc, rad=rad)
            np.savez(vortexfile, **vortex)
        else:
            vortex = np.load(vortexfile)
        figfile = root_path + 'Fig_' + str(noise_level) + '.eps'
        compare_figures(vortex, info, vx, vy, file=figfile)
        detection_rate(vortex, info, noise_level)
