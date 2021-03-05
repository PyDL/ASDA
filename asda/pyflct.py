#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 5 17:58:29 2017

Name: pyflct.py

Discription: Python wrapper for FLCT code written in C from Fisher & Welsch
             2008. You can download the original C code from the following:
             http://cgem.ssl.berkeley.edu/cgi-bin/cgem/FLCT/home

             Before a proper run of this program, you need first install
             the FLCT libraries. Extract the downloaded C source files, go
             to the fold. Then check source/README-install.txt
             and Makefile to find out how to install the FLCT libraries
             properly.

Inputs:
    data1 - image data at time T
    data2 - image data at time T+dT
    deltat - time difference between two images
    deltas - pixel size of each image, units of velocities will be
             based on deltas/deltat
    sigma - pixel width of the Gausian filter. Default is 10 according
            to [Louis et al. 2015, Solar Phys. 290, 1135], which found
            the optimal value of sigma for their simulated intensity
            map ranges from 10 to 15. If sigma is set to 0, only the
            overall correlation between two input images will be
            calculated.
    infile - name of file which will be generated storing data1 and data2
    outfile - name of file which stores the FLCT velocity field
    thresh - Do not compute the velocity at a given pixel if the average
             absolute value between the 2 images at that location is less
             than thresh.
    kr - from 0 to 1. Perform gaussian, low pass filtering on the
         sub-images that are used to construct the cross-correlation
         function. The value of kr is expressed in units of the maximum
         wavenumber (Nyquist frequency) in each direction of the
         sub-image. This option is most useful when the images contain
         significant amounts of uncorrelated, pixel-to-pixel noise-like
         structure. Empirically, values of kr in the range of 0.2 to 0.5
         seem to be most useful, with lower values resulting in stronger
         filtering.
    skip - if is not None, the program will only compute the velocity
           every N pixels in both the x and y direction, where N is
           the value of skip.
    xoff, yoff - only valid when skip is not None. xoff and yoff
                 are the offset in x and y direction where the compuation
                 starts.
    interp - only valid when skip is not None. If true, interpolation
             at pixels skipped will be done.
    pc - If set, then the input images are assumed to be in Plate Carree
         coordinates (uniformly spaced in longitude and latitude). This
         is useful when e.g. input images are SHARP magnetic field data.
    bc - If set, bias correction will be turned on (new feature in 1.06)
    latmin, latmax - minimum and maximum latitude. Only valid if pc is
                     set. In units of radian.
    quiet - If set, no non-error output will be shown.
Outputs:
    a tupe in form of (vx, vy, vm)
    where, vx is the velocity field in x direction;
           vy is the velocity field in y direction;
           vm is the velocity field mask. Pixels with vm value of 0 have
               not been included when calculating the velocity field.
               Pixels with vm value of 0.5 have interpolated velocity
               field.

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv3'
__version__ = '2.0'
__date__ = '2019/11/27'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'

import numpy as np
import struct
import os


def vcimageout(data, filename='./testin.dat'):
    '''
    Adapted from vcimage1/2/3out.pro in Fisher & Welsch 2008.
    Input: data - tuple containing all image data to be stored in filename
                  It must be in the following format:
                  data = (data1, data2 ...)
                  where, data1, data2 ... are image data at different times

           filename - name of file that will be generated
    Output: Return - None
            A new file with name of filename will be generated containing
            data, which will be used for the FLCT C code.
    '''

    # Perform size and shape check
    num = len(data)
    shapes = []
    sizes = []
    for i in range(num):
        array = data[i]
        array = np.array(array, dtype=np.float32)
        shapes.append(array.shape)
        sizes.append(array.size)
    for v in sizes:
        if v != sizes[0]:
            print('vcimage2out: dimensions or ranks of data ' +
                  'do not match')
            return None
    for v in shapes:
        if len(v) != 2 and sizes[0] > 1:
            print('vcimage2out: input array is not 2d')
            return None
    # initial integer ID for a "vel_ccor" i/o file
    vcid = int(2136967593).to_bytes(4, 'big')
    if sizes[0] > 1:
        nx = int(data[0].shape[0])
        ny = int(data[0].shape[1])
    else:
        nx = int(1)
        ny = int(1)
    nx = nx.to_bytes(4, 'big')
    ny = ny.to_bytes(4, 'big')
    f = open(filename, 'wb')
    f.write(vcid)
    f.write(nx)
    f.write(ny)
    for i in range(num):
        array = data[i]
        for value in np.nditer(array, order='F'):
            v = struct.pack('>f', value)
            f.write(v)

    f.close()


def vcimagein(filename='testout.dat'):
    '''
    Adapted from vcimage1/2/3out.pro in Fisher & Welsch 2008.
    Input: filename - name of C binary file
    Output: Return - a list containing arrays of data
            In the case of reading velocity field generated by FLCT:
            vx - velocity field in x direction
            vy - velocity field in y direction
            vm - velocity field mask. Pixels with vm value of 0 have not been
                 included when calculating the velocity field
    '''
    f = open(filename, 'rb')
    vcid = struct.unpack('>i', f.read(4))[0]
    if vcid != 2136967593:
        print('Input file is not a vel_coor i/o file')
        f.close()
        return None
    nx = struct.unpack('>i', f.read(4))[0]
    ny = struct.unpack('>i', f.read(4))[0]
    data = f.read()
    f.close()
    # calculate number of files
    num = int(len(data) / (4. * nx * ny))
    f.close()
    array = ()
    for k in range(num):
        dust = np.zeros((nx, ny), dtype=np.float32)
        offset = nx * ny * k * 4
        idx = offset
        # In the case when sigma is set to zero using FLCT
        if nx == 1 and ny == 1:
            v = struct.unpack('>f', data[idx:idx+4])[0]
            array = array + (v,)
        else:
            it = np.nditer(dust, flags=["multi_index"],
                           op_flags=["readwrite"], order='F')
            while not it.finished:
                v = struct.unpack('>f', data[idx:idx+4])[0]
                dust[it.multi_index] = v
                it.iternext()
                idx = idx + 4
            array = array + (dust,)

    return array


def flct(data1, data2, deltat=1, deltas=1, sigma=10, infile="testin.dat",
         outfile="testout.dat", thresh=None, kr=None, skip=None, xoff=0,
         yoff=0, interp=False, pc=False, bc=False, latmin=0, latmax=0.2,
         quiet=False):
    '''
    Main function of the Python wrapper pyflct for the FLCT C code
    Inputs:
        data1 - image data at time T
        data2 - image data at time T+dT
        deltat - time difference between two images
        deltas - pixel size of each image, units of velocities will be
                 based on deltas/deltat
        sigma - pixel width of the Gausian filter. Default is 10 according
                to [Louis et al. 2015, Solar Phys. 290, 1135], which found
                the optimal value of sigma for their simulated intensity
                map ranges from 10 to 15. If sigma is set to 0, only the
                overall correlation between two input images will be
                calculated.
        infile - name of file which will be generated storing data1 and data2
        outfile - name of file which stores the FLCT velocity field
        thresh - Do not compute the velocity at a given pixel if the average
                 absolute value between the 2 images at that location is less
                 than thresh.
        kr - from 0 to 1. Perform gaussian, low pass filtering on the
             sub-images that are used to construct the cross-correlation
             function. The value of kr is expressed in units of the maximum
             wavenumber (Nyquist frequency) in each direction of the
             sub-image. This option is most useful when the images contain
             significant amounts of uncorrelated, pixel-to-pixel noise-like
             structure. Empirically, values of kr in the range of 0.2 to 0.5
             seem to be most useful, with lower values resulting in stronger
             filtering.
        skip - if is not None, the program will only compute the velocity
               every N pixels in both the x and y direction, where N is
               the value of skip.
        xoff, yoff - only valid when skip is not None. xoff and yoff
                     are the offset in x and y direction where the compuation
                     starts.
        interp - only valid when skip is not None. If true, interpolation
                 at pixels skipped will be done.
        pc - If set, then the input images are assumed to be in Plate Carree
             coordinates (uniformly spaced in longitude and latitude). This
             is useful when e.g. input images are SHARP magnetic field data.
        bc - If set, bias correction will be turned on (new feature in 1.06)
        latmin, latmax - minimum and maximum latitude. Only valid if pc is
                         set. In units of radian.
        quiet - If set, no non-error output will be shown.
    Outputs:
        a tupe in form of (vx, vy, vm)
        where, vx is the velocity field in x direction;
               vy is the velocity field in y direction;
               vm is the velocity field mask. Pixels with vm value of 0 have
                   not been included when calculating the velocity field.
                   Pixels with vm value of 0.5 have interpolated velocity
                   field.
    '''
    data = (data1, data2)
    infile = str(infile)
    outfile = str(outfile)
    vcimageout(data, infile)
    command = "flct " + infile + " "
    command = command + str(outfile)
    deltat = "{:.6f}".format(float(deltat))
    deltas = "{:.6f}".format(float(deltas))
    sigma = "{:.6f}".format(float(sigma))
    command = command + " " + deltat + " " + deltas + " " + sigma
    if thresh is not None:
        thresh = float(thresh)
        if thresh < 0:
            print("Threshold with value less than 0 will not be used.")
        else:
            thresh = "{:.6f}".format(thresh)
            command = command + " -t " + thresh
    if kr is not None:
        kr = float(kr)
        if kr < 0.0 or kr > 1.0:
            print("Kr must range from 0 to 1.")
        else:
            kr = "{:.6f}".format(kr)
            command = command + " -k " + kr
    if skip is not None:
        skip = int(skip)
        if skip <= 0:
            print("Skip must be a positive number")
        else:
            xoff = int(xoff)
            yoff = int(yoff)
            skip = "{:d}".format(skip)
            xoff = "{:d}".format(xoff)
            yoff = "{:d}".format(yoff)
            command = command + " -s N" + skip + "p" + xoff + "q" + yoff
            if interp:
                command = command + "i"
    if pc:
        latmin = "{:.6f}".format(float(latmin))
        latmax = "{:.6f}".format(float(latmax))
        command = command + " -pc " + latmin + " " + latmax
    if bc:
        command = command + " -bc"
    if quiet:
        command = command + " -q"
    os.system(command)
    result = vcimagein(outfile)

    return result


if __name__ == "__main__":
    # The following is a test
    a = np.random.rand(101, 101)
    b = np.roll(a, 5, axis=0)
    b = np.roll(b, -3, axis=1)
    result = flct(a, b, 1, 1, 15)
