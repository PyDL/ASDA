# ASDA
Automatic Swirl Detection Algorithms

## Dependencies:
**Python 3** with libraries including numpy, scipy, getopt, sys, matplotlib, random, os, subprocess, multiprocessing, mahotas, mpi4py (optional)</br>
**pyflct**: https://github.com/PyDL/pyflct

## Description of Files (More information can be found in each file):
**vortex.py**: Main programm of the implication of the swirl detection algorithms</br>
**gamma_values_mpi.py**: MPI version of the vortex.gamma_values() function</br>
**points_in_poly.py**: return all points within a polygon</br>
**swirl_lifetime.py**: using different methods to give labels to all swirls in order to estimate their lifetimes</br>
**lamb_oseen.py**: Object of a lamb_oseen vortex</br>
**test_synthetic.py**: Main program generating and testing a series of synthetic data (see reference)</br>

## Example of Usage:
You can also find the following steps from line 249 in `est_synthetic.py`.
Suppose you have two succesive 2d images in **(x, y)** order: data0 and data1</br>
1. import neccessary libraries, including:
```python
from pyflct import flct, vcimagein
from vortex import gamma_values, center_edge, vortex_property
```
2. you need to use the pyflct package to estimate the velocity field connecting the above two images: 
`vx, vy, vm = flct(data0, data1, 1.0, 1.0, 10, outfile='vel.dat')`. Please notice that, vx, vy and vm are also in *(x, y)** order. Here, vx and vy are the velocity field. Usually, vm are not necessary.</br>
3. calculate gamma1 and gamma2 values (see the reference) with `gamma1, gamma2 = gamma_values(vx, vy, factor=1)`. Alternatively, you may use `gamma_values_mpi.py` to make the calculation MUCH faster with multiple CPUs. You may specify factor as an integer greater than 1 to make preciser detection of vortex centers. But, this can be very costy and is not neccessary for high-resolution data.</br>
4. perform the detection of vortices using `center, edge, point, peak, radius = center_edge(gamma1, gamma2, factor=1)`. center is a list containing the pixel location of all vortices in the image. edge is a list of the edges of all vortices. point is a list of all points within vortices. peak is a list of the peak gamma1 values of all vortices. radius is a list of the effective radii of all vortices.</br>
5. use `ve, vr, vc, ia = vortex_property(center, edge, points, vx, vy, data0)` to calculate the expanding, rotating, center speeds of above vortices. ia is the average intensity from data0 of all points within each vortex.</br>
6. **Notice**: radius, ve, vr and vc calculated above are in units of 1. Suppose for data0 and data1, the pixel size is *ds* (in units of actual physical units such as Mm, km, m...) and the time difference of *dt* (in units of second, minute, hour...), then you should use `radius * ds` and `ve * ds / dt`, `vr * ds / dt`, `vc * ds / dt` as your final results.



## Credit:
Liu, J., Nelson, C, Erdelyi, R, Automated Swirl Detection Algorithm (ASDA) and Its Application to Simulation and Observational Data, 2018 (https://arxiv.org/abs/1804.02931)

