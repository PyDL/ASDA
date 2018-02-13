#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:36:09 2017

Copy from Ulf Aslak's answer in the following link:
    https://stackoverflow.com/questions/21339448/how-to-get
    -list-of-points-inside-a-polygon-in-python
author: Jiajia Liu @ University of Sheffield
"""

import numpy as np
import mahotas


def points_in_poly(poly):
    """Return polygon as grid of points inside polygon.

    Input : poly (list of lists)
    Output : output (list of lists)
    """
    xs, ys = zip(*poly)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]

    X = round(maxx - minx) + 1
    Y = round(maxy - miny) + 1

    grid = np.zeros((X, Y), dtype=np.int8)
    mahotas.polygon.fill_polygon(newPoly, grid)

    return [(x + minx, y + miny) for (x, y) in zip(*np.nonzero(grid))]