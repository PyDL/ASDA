#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:36:09 2017

Inspired by Ulf Aslak's answer in the following link:
    https://stackoverflow.com/questions/21339448/how-to-get
    -list-of-points-inside-a-polygon-in-python
author: Jiajia Liu @ University of Sheffield

Nov 2019: Use scikit-image instead of mahotas, result is slightly different
"""

import numpy as np
from skimage import measure


def points_in_poly(poly):
    """
    Return polygon as grid of points inside polygon. Only works for polygons
    defined with points which are all integers

    Parameters
    ----------
    poly : `list` or `numpy.ndarray`
        n x 2 list, defines all points at the edge of a polygon

    Returns
    -------
        `list`
        n x 2 array, all points within the polygon

    """
    if np.shape(poly)[1] != 2:
        raise ValueError("Polygon must be defined as a n x 2 array!")

    # convert to integers
    poly = np.array(poly, dtype=int).tolist()

    xs, ys = zip(*poly)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    # New polygon with the staring point as [0, 0]
    newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]
    mask = measure.grid_points_in_poly((round(maxx - minx) + 1,
                                        round(maxy - miny) + 1), newPoly)
    # all points in polygon
    points = [[x + minx, y + miny] for (x, y) in zip(*np.nonzero(mask))]

    # add edge points if missing
    for p in poly:
        if p not in points:
            points.append(p)

    return points