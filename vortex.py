# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 2017

Name: vortex.py

Discription: Vortex detection based on Graftieaux et al. 2001

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2017, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv2'
__version__ = '1.00'
__date__ = '2017/12/07'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from points_in_poly import points_in_poly
from scipy.interpolate import interp2d
from itertools import product
from skimage import measure


def reform2d(array, factor=1):
    """
    Reform a 2d array by a given factor

    Parameters
    ----------
    array : `numpy.ndarray`
        2d array to be reformed

    factor : `int`
        The array is going to be magnified by the factor. Default is 1.

    Returns
    -------
        `numpy.ndarray`
        reformed array

    """
    if not isinstance(factor, int):
        raise ValueError("Parameter 'factor' must be an integer!")

    if len(np.shape(array)) != 2:
        raise ValueError("Input array must be 2d!")

    if factor > 1:
        congridx = interp2d(np.arange(0, array.shape[0]),
                            np.arange(0, array.shape[1]), array.T)
        array = congridx(np.arange(0, array.shape[0], 1/factor),
                         np.arange(0, array.shape[1], 1/factor)).T

    return array


def remove_duplicate(edge):
    """
    Remove duplicated points in a the edge of a polygon

    Parameters
    ----------
    edge : `list` or `numpy.ndarray`
        n x 2 list, defines all points at the edge of a polygon

    Returns
    -------
        `list`
        same as edge, but with duplicated points removed
    """

    shape = np.shape(edge)
    if shape[1] != 2:
        raise ValueError("Polygon must be defined as a n x 2 array!")

    new_edge = []
    for i in range(shape[0]):
        p = edge[i]
        if not isinstance(p, list):
            p = p.tolist()
        if p not in new_edge:
            new_edge.append(p)
    
    return new_edge


def gamma_values(vx, vy, r=3, factor=1, nthreads=1):
    '''
    Purpose: calculate gamma1 and gamma2 values of velocity field vx and vy
             Fomulars can be found in Graftieaux et al. 2001
    Inputs:
        vx - velocity field in x direction
        vy - velocity field in y direction
        r - maximum distance of neighbour points from target point
        factor - default is 2. Magnify the original data to find sub-grid
                 vortex center and boundary
    Outputs:
        gamma - tuple in form of (gamma1, gamma2), where gamma1 is useful in finding
              vortex centers and gamma2 is useful in finding vortex edges
    '''

    vx = np.array(vx, dtype=np.float32)
    vy = np.array(vy, dtype=np.float32)
    if vx.shape != vy.shape:
        print("Velocity field vx and vy do not match!")
        return None
    r = int(r)
    factor = int(factor)
    if factor > 1:
        vx = reform2d(vx, factor=factor)
        vy = reform2d(vy, factor=factor)
    nx = vx.shape[0]
    ny = vx.shape[1]

    gamma = np.array([np.zeros_like(vx),
                      np.zeros_like(vy)])
    # pm vectors, see equation (8) in Graftieaux et al. 2001 or Equation
    # (1) in Liu et al. 2019
    pm = np.array([[i, j]
                    for i in np.arange(-r, r + 1)
                    for j in np.arange(-r, r + 1)], dtype=float)

    # mode of vector pm
    pnorm = np.linalg.norm(pm, axis=1)

    # Number of points in the concerned region
    N = (2 * r + 1) ** 2

    # Create index array
    index = np.array([[i, j]
                        for i in np.arange(r, ny - r)
                        for j in np.arange(r, nx - r)])

    # Transpose index
    index = index.T

    # Generate velocity field
    vel = gen_vel(vx, vy, index[1], index[0], r=r)

    # Iterate over the array gamma
    for d, (i, j) in enumerate(product(np.arange(r, ny - r, 1),
                                        np.arange(r, nx - r, 1))):

        gamma[0, j, i], gamma[1, j, i] = calc_gamma(pm, vel[..., d], pnorm, N)

    return gamma


def center_edge(gamma1, gamma2, factor=1, rmin=4, gamma_min=0.89):
    '''
    Find vortices from gamma1, and gamma2
    Output:
        center: center location of vortices
        edge: edge location of vortices
        points: all points within vortices
        peak: maximum/minimum gamma1 value in vortices
        radius: equivalent radius of vortices
        All in pixel coordinates
    '''
    # ------------ deprecated ------------------------------------------------
    # matplotlib.interactive(False)
    # plt.subplots()
    # cs = plt.contour(gamma2.T, levels=[-2 / np.pi, 2 / np.pi])
    # plt.close()

    edge = ()
    center = ()
    points = ()
    peak = ()
    radius = ()
    # for i in range(len(cs.collections)):
    #     cnts = cs.collections[i].get_paths()
    #     for c in cnts:
    #         v = np.rint(c.vertices).tolist()
    cs = np.array(measure.find_contours(gamma2, -2 / np.pi))
    cs_pos = np.array(measure.find_contours(gamma2, 2 / np.pi))
    if len(cs) == 0:
        cs = cs_pos
    elif len(cs_pos) != 0:
        cs = np.append(cs, cs_pos, 0)
    for i in range(np.shape(cs)[0]):
        v = np.rint(cs[i])
        v = remove_duplicate(v)
        ps = points_in_poly(v)
        dust = []
        for p in ps:
            dust.append(gamma1[int(p[0]), int(p[1])])
        if len(dust) > 1:
            re = np.sqrt(np.array(ps).shape[0]/np.pi) / factor
            if np.max(np.fabs(dust)) >= gamma_min and re >= rmin :
                # allow some error around 0.9
                # vortex with radius less than 4 pixels is not reliable
                idx = np.where(np.fabs(dust) == np.max(np.fabs(dust)))
                idx = idx[0][0]
                center = center + (np.array(ps[idx])/factor, )
                edge = edge + (np.array(v)/factor, )
                points = points + (np.array(ps)/factor, )
                peak = peak + (dust[idx], )
                radius = radius + (re, )
#    edge = np.array(edges, dtype=int)
#    center = np.array(centers, dtype=int)
    return (center, edge, points, peak, radius)


def vortex_property(centers, edges, points, vx, vy, image=None):
    '''
    Calculate expanding, rotational speed, equivalent radius and average
        intensity of given swirls.
    Output:
        ve: expanding speed, pixel/frame
        vr: rotational speed, pixel/frame
        vc: velocity of the center, pixel/frame
        ia: average the observation values (intensity or magnetic field)
            within the vortices
    '''
    vx = np.array(vx)
    vy = np.array(vy)
    n_swirl = len(centers)
    ve = ()
    vr = ()
    vc = ()
    ia = ()

    for i in range(n_swirl):
        cen = centers[i]
        edg = edges[i]
        pnt = np.array(points[i], dtype=int)
        x0 = int(round(cen[0]))
        y0 = int(round(cen[1]))
        vcen = [vx[x0, y0], vy[x0, y0]]
        vc = vc + (vcen, )
        if image is not None:
            image = np.array(image)
            value = 0
            for pos in pnt:
                value = value + image[pos[0], pos[1]]
            value = value * 1.0 / pnt.shape[0]
        else:
            value = None
        ia = ia + (value, )
        ve0 = []
        vr0 = []
        for j in range(edg.shape[0]):
            idx = [edg[j][0], edg[j][1]]
            pm = [idx[0]-cen[0], idx[1]-cen[1]]
            tn = [cen[1]-idx[1], idx[0]-cen[0]]
            idx = np.array(idx, dtype=int)
            v = [vx[idx[0], idx[1]], vy[idx[0], idx[1]]]
            ve0.append(np.dot(v, pm)/np.linalg.norm(pm))
            vr0.append(np.dot(v, tn)/np.linalg.norm(tn))
        ve = ve + (np.nanmean(ve0), )
        vr = vr + (np.nanmean(vr0), )

    return (ve, vr, vc, ia)


def gen_vel(vx, vy, i, j, r=3):
    """
    Given a point [i, j], generate a velocity field which contains
    a region with a size of (2r+1) x (2r+1) centered at [i, j] from
    the original velocity field vx and vy.

    Parameters
    ----------
    i : `int`
        first dimension of the pixel position of a target point.
    j : `int`
        second dimension of the pixel position of a target point.

    Returns:
    -------
        `numpy.ndarray`
        the first dimension is a velocity field which contains a
        region with a size of (2r+1) x (2r+1) centered at [i, j] from
        the original velocity field vx and vy.
        the second dimension is similar as the first dimension, but
        with the mean velocity field substracted from the original
        velocity field.
    """

    vel = np.array([[vx[i + im, j + jm], vy[i + im, j + jm]]
                    for im in np.arange(-r, r + 1)
                    for jm in np.arange(-r, r + 1)])

    return np.array([vel, vel - vel.mean(axis=0)])

def calc_gamma(pm, vel, pnorm, N):
    """
    Calculate Gamma values, see equation (8) in Graftieaux et al. 2001
    or Equation (1) in Liu et al. 2019

    Parameters
    ----------
        pm : `numpy.ndarray`
            vector from point p to point m
        vel : `numpy.ndarray`
            velocity vector
        pnorm : `numpy.ndarray`
            mode of pm
        N : `int`
            number of points

    Returns
    -------
        `float`
        calculated gamma values for velocity vector vel
    """

    cross = np.cross(pm, vel)
    vel_norm = np.linalg.norm(vel, axis=2)
    sint = cross / (pnorm * vel_norm + 1e-10)

    return np.nansum(sint, axis=1) / N