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
__version__ = '1.04'  # consistent with the version of the C code
__date__ = '2017/12/07'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from points_in_poly import points_in_poly
from scipy.interpolate import interp2d


def reform2d(array, factor=1):

    array = np.array(array)
    factor = int(factor)
    if factor > 1:
        nx = array.shape[0]
        ny = array.shape[1]
        x = np.arange(0, nx)
        y = np.arange(0, ny)
        xnew = np.arange(0, nx, 1./factor)
        ynew = np.arange(0, ny, 1./factor)
        congridx = interp2d(x, y, array.T)
        array = congridx(xnew, ynew).T

    return(array)


def gamma_values(vx, vy, r=3, factor=1):
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
    for i in np.arange(r, nx-r, 1):
        for j in np.arange(r, ny-r, 1):
            vel = []
            for im in neighbour:
                for jm in neighbour:
                    vel.append([vx[i+im, j+jm], vy[i+im, j+jm]])
            vel = np.array(vel)
            cross = np.cross(pm, vel, axis=1)
            # prevent zero divided error
            sint = cross / (pnorm * np.linalg.norm(vel, axis=1) + 1e-10)
            gamma1[i, j] = np.nansum(sint) / N
            vel2 = vel - vel.mean(axis=0)
            cross = np.cross(pm, vel2, axis=1)
            sint = cross / (pnorm * np.linalg.norm(vel2, axis=1) + 1e-10)
            gamma2[i, j] = np.nansum(sint) / N

    return (gamma1, gamma2)


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
    matplotlib.interactive(False)
    plt.subplots()
    cs = plt.contour(gamma2.T, levels=[-2 / np.pi, 2 / np.pi])
    plt.close()
    edge = ()
    center = ()
    points = ()
    peak = ()
    radius = ()
    for i in range(len(cs.collections)):
        cnts = cs.collections[i].get_paths()
        for c in cnts:
            v = np.rint(c.vertices).tolist()
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


class Freestream:
    def __init__(self, u_inf=1.0, alpha=0.0):
        """Sets the freestream conditions.
        Arguments
        ---------
        u_inf -- Farfield speed (default 1.0).
        alpha -- Angle of attack in degrees (default 0.0).
        """
        self.u_inf = u_inf
        self.alpha = alpha*np.pi/180


def generate_vortex(ny=50, nx=50, strength=5.0):
    """Returns the velocity field generated by a vortex.
    """
    x_start, x_end = -2.0, 2.0
    y_start, y_end = -2.0, 2.0
    x = np.linspace(x_start, x_end, nx)
    y = np.linspace(y_start, y_end, ny)
    X, Y = np.meshgrid(x, y)
    xc = 0
    yc = 0
    u = + strength/(2*np.pi)*(Y-yc)/((X-xc)**2+(Y-yc)**2)
    v = - strength/(2*np.pi)*(X-xc)/((X-xc)**2+(Y-yc)**2)
    freestream = Freestream(2.0, 0.0)
    u += freestream.u_inf * X * np.cos(freestream.alpha)
    v += freestream.u_inf * Y * np.sin(freestream.alpha)
    return (X, Y, u, v)


if __name__ == '__main__':
    x, y, vx, vy = generate_vortex()

    # because python is row-major
    gamma = gamma_values(vx.transpose(), vy.transpose())
