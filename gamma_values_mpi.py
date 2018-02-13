#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 16:58:59 2017

Name: gamma_values_mpi.py

Purpose: MPI version of vortex.gamma_values. Please note it may use lots of
         memories when the input image is large.

Input: f: velocity field file generated by pyflct.py
       r - maximum distance of neighbour points from target point
       a - default is 2. Magnify the original data to find sub-grid
                 vortex center and boundary
       o: name of output file storing calculated gamma values


@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv2'
__version__ = '1.00'  # consistent with the version of the C code
__date__ = 2017/12/19
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'


import numpy as np
from scipy.interpolate import interp2d
import getopt
import sys
from mpi4py import MPI
from pyflct import vcimagein, vcimageout
from vortex import gamma_values, reform2d

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

r = 3
factor = 1
ofile = 'tmp.dat'
file = 'tmp.dat'
opts, args= getopt.getopt(sys.argv[1:], "f:r:a:o:s")

for opt, arg in opts:
    if opt == '-f':
        file = str(arg)
    if opt == '-r':
        r = int(float(arg))
    if opt == '-a':
        factor = int(float(arg))
    if opt == '-o':
        ofile = str(arg)
vx = []
vy = []
if rank == 0:
    vx, vy, vm = vcimagein(file)
    vx = np.array(vx, dtype=np.float32)
    vy = np.array(vy, dtype=np.float32)
    if vx.shape != vy.shape:
        print("Velocity field vx and vy do not match!")
        sys.exit()
    if factor > 1:
        vx = reform2d(vx, factor=factor)
        vy = reform2d(vy, factor=factor)

vx = comm.bcast(vx, root=0)
vy = comm.bcast(vy, root=0)
nx = vx.shape[0]
ny = vx.shape[1]

gamma1 = np.zeros_like(vx)
gamma2 = np.zeros_like(vx)
neighbour = np.arange(-r, r+1)
pm = []
for im in neighbour:
    for jm in neighbour:
        pm.append([im, jm])
pm = np.array(pm, dtype=float)
pnorm = np.linalg.norm(pm, axis=1)

N = (2 * r + 1) ** 2

each = (nx-2*r) / int(size)
if each < 2:
    print("Input image is small, no need to use MPI. \n"
          "Using sequencial calculation...")
    if rank == 0:
        gamma = gamma_values(vx, vy, r=r, factor=1)
else:
    lower = int(each * rank + r)
    upper = int(each * (rank + 1) + r)
    if upper > nx-r:
        upper = nx-r
    for i in np.arange(lower, upper, 1):
        for j in np.arange(r, ny-r, 1):
            vel = []
            for im in neighbour:
                for jm in neighbour:
                    vel.append([vx[i+im, j+jm], vy[i+im, j+jm]])
            vel = np.array(vel)
            cross = np.cross(pm, vel, axis=1)
            sint = cross / (pnorm * np.linalg.norm(vel, axis=1) + 1e-10)
            gamma1[i, j] = np.nansum(sint) / N
            vel2 = vel - vel.mean(axis=0)
            cross = np.cross(pm, vel2, axis=1)
            sint = cross / (pnorm * np.linalg.norm(vel2, axis=1) + 1e-10)
            gamma2[i, j] = np.nansum(sint) / N

    if rank != 0:
        comm.send(gamma1, dest=0, tag=rank * 10 + 1)
        comm.send(gamma2, dest=0, tag=rank * 100 + 1)
    else:
        for i in np.arange(size - 1) + 1:
            dust = comm.recv(source=i, tag=i * 10 + 1)
            dust2 = comm.recv(source=i, tag=i * 100 + 1)
            lower = int(each * i + r)
            upper = int(each * (i + 1) + r)
            if upper > nx-r:
                upper = nx-r
            gamma1[lower:upper, :] = dust[lower:upper, :]
            gamma2[lower:upper, :] = dust2[lower:upper, :]
        vcimageout((gamma1, gamma2), ofile)
