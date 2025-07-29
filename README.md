## Changelog

**July 2025, v2.2.2 release. The changes are:</br>**

**November 2023: v2.2.1 released. The changes are:</br>**
A bug in center_edge() is now fixed. It appeared when only 1 positive or negative swirl was detected. 

**June 2021: v2.2 released. The changes are:</br>**
A new function gamma_values_parallel is added to vortex.py. It uses multiprocessing to do the parallelisation. For the data in used in demo.py, the parallel version is ~2 times faster than the v2.1 on my Macbook Pro.</br>

**March 5 2021: v2.1 released. The changes are:</br>**
    1. add save_vortex and read_vortex functions to vortex.py, supports both npz and hdf5 (could be read in Matlab and IDL) formats.</br>
    2. pyflct is now included in ASDA package. **You need to install FLCT C code before pyflct works**.</br>
    3. new setup.py file, you can now install ASDA into your python environment.</br>

**November 29 2019: v2.0 released. There are 2 significant changes:</br>**
    1. the time consumed to calculate Gamma values is now 3 times shorter than v1.0. Thanks to Nobert Gyenge @gyenge. The speed is now very close to the MPI version of v1.0.</br>
    2. In points_in_poly.py, we have changed the dependency from mahotas to scikit-image. This change results in the detected radius changing by several percent. We demonstrate this is normal.</br>
    3. In vortex.py, we have changed the tool for finding contours from matplotlib to scikit-image. These 2 tools give the same result for contours, but do different interpolations. Now we convert the found contours to integers and introduce a new funtion to remove all duplicated points in the found contours. We demonstrate that, the above change don't change the number, position and center of swirls detected. But have little effect on the radius, rotating speed, and average observational value of swirls. The influence on the expanding/shrinking speed could sometimes be large considering expanding/shrinking speeds are usually very small.</br>
    4. We also tested the above change with artificially generated Lamb Oseen vortices, the above change only have very little influence on the detected radius (because of change 2). All other properties of the detected vortices keep unchanged.</br>
  
## Cite:
Liu, J., Nelson, C, Erdelyi, R, Automated Swirl Detection Algorithm (ASDA) and Its Application to Simulation and Observational Data, ApJ, 872, 22, 2019 (https://iopscience.iop.org/article/10.3847/1538-4357/aabd34/meta)

## Please contact the authors and ask for permissions before using any part of these code.
Email: ~~jj.liu@sheffield.ac.uk~~ jiajialiu@ustc.edu.cn

# ASDA
Automatic Swirl Detection Algorithms

## System Requirements
### OS Requirements
ASDA can be run on Windows, Mac OSX or Linux systems with Python 3 and the following dependencies installed.

### Dependencies:
**Python 3** with libraries including numpy, scipy, getopt, sys, matplotlib, random, os, subprocess, multiprocessing, scikit-image, mpi4py</br>
**pyflct**: https://github.com/PyDL/pyflct </br>
**Python management softwares including Anaconda or Virtualenv are recommended**

## Hardware Requirements:
ASDA requires a standard computer with enough CPU and computation power depending on the dataset used. To be able to use the parallel functions, a multi-core CPU supporting the MPI libraries will be needed.

## Installation Guide:
If you are using Anaconda, install dependencies using ``conda install`` before installing ASDA

run the following codes in your terminal to install ASDA:
```bash
git clone https://github.com/PyDL/ASDA
cd ASDA
pip install .
```
ASDA will be then installed in your default Python environment.

## Description of Files (More information can be found in each file):
**vortex.py**: Main programm of the implication of the swirl detection algorithms</br>
**gamma_values_mpi.py**: MPI version of the vortex.gamma_values() function</br>
**points_in_poly.py**: return all points within a polygon</br>
**swirl_lifetime.py**: using different methods to give labels to all swirls in order to estimate their lifetimes</br>
**lamb_oseen.py**: Object of a lamb_oseen vortex</br>
**test_synthetic.py**: Main program generating and testing a series of synthetic data (see reference)</br>
**correct.npz**: correct swirl detection result</br>
**correct_v1.npz**: correct swirl detection result from version 1, for comparison purpose</br>
**setup.py**: setup file used for pip</br>

## Instructions for Use:
You can also find the following steps from line 249 in `test_synthetic.py`.
Suppose you have two succesive 2d images in **(x, y)** order: data0 and data1</br>
1. import neccessary libraries, including:
```python
from asda.pyflct import flct, vcimagein
from asda.vortex import gamma_values, center_edge, vortex_property
```
2. you need to use the pyflct package to estimate the velocity field connecting the above two images: 
`vx, vy, vm = flct(data0, data1, 1.0, 1.0, 10, outfile='vel.dat')`. Please notice that, vx, vy and vm are also in **(x, y)** order. Here, vx and vy are the velocity field. Usually, vm are not necessary.</br>
1. calculate gamma1 and gamma2 values (see the reference) with `gamma1, gamma2 = gamma_values(vx, vy, factor=1)`. Alternatively, you may use `gamma_values_mpi.py` to make the calculation MUCH faster with multiple CPUs. You may specify factor as an integer greater than 1 to make preciser detection of vortex centers. But, this can be very costy and is not neccessary for high-resolution data.</br>
2. perform the detection of vortices using `center, edge, point, peak, radius = center_edge(gamma1, gamma2, factor=1)`. center is a list containing the pixel location of all vortices in the image. edge is a list of the edges of all vortices. point is a list of all points within vortices. peak is a list of the peak gamma1 values of all vortices. radius is a list of the effective radii of all vortices.</br>
3. use `ve, vr, vc, ia = vortex_property(center, edge, points, vx, vy, data0)` to calculate the expanding, rotating, center speeds of above vortices. ia is the average intensity from data0 of all points within each vortex.</br>
4. **Notice**: radius, ve, vr and vc calculated above are in units of 1. Suppose for data0 and data1, the pixel size is *ds* (in units of actual physical units such as Mm, km, m...) and the time difference of *dt* (in units of second, minute, hour...), then you should use `radius * ds` and `ve * ds / dt`, `vr * ds / dt`, `vc * ds / dt` as your final results.

## Demo
A demo **demo.py** is available with the demo data **demo_data.sav**:
1. To run the demo, `cd ASDA` and run `python demo.py`
2. The demo data consists of the following 4 variables: data0 (SOT Ca II observation at 2007-03-05T05:48:06.737), data1 (SOT Ca II observation at 2007-03-05T05:48:13.138), ds (pixel size of the observations), and dt (time difference in seconds between data0 and data1)
### Expected Output
After running the code, you will see 3 files as the output: **vel_demo.dat** (binary file storing the calculated velocity field, 6MB), **gamma_demo.dat** (binary file storing gamma1 and gamma2, 6MB), and **vortex_demo.npz** (numpy file storing the information of all deteced vortices, 192.9 kB). All the differences printed out should be 0.</br>
</br>
Use `vortex = dict(np.load('vortex_demo.npz'))`, you should see the variable `vortex` stores the center, edge, points, peak (gamma1 value), radius, ve (average expanding/shrinking velocity), vr (average rotating speed), vc (speed of center), and ia (average observation intensity) for **52** detected swirls. You can compare these results with the correct detection result stored in **correct.npz**
### Expected Running Time
Once finished, the command line will give the time consumption of the demo code, which should be ~20 seconds on an Intel I7 4.20 GHz CPU.


