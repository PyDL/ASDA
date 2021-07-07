#!/usr/bin/env python
# coding=utf-8
# by Jiajia Liu @ Queen's University Belfast
# March 2021

from setuptools import setup, find_packages
import glob
import os


def main():
    setup(
        name="ASDA",
        python_requires='>3.5.0',
        version="2.2",
        author="Jiajia Liu",
        author_email="j.liu@qub.ac.uk",
        description=("Automated Swirl Detection Algorithm"),
        license="GPLv3",
        keywords="ASDA",
        url="https://github.com/PyDL/ASDA",
        packages=find_packages(where='.', exclude=(), include=('*',)),
        py_modules=get_py_modules(),

        # dependencies
        install_requires=[
            'numpy',
            'scipy',
            'scikit-image',
            'mpi4py',
            'matplotlib',
            'h5py'
        ],

        classifiers=[
            "Development Status :: 2.2 - Release",
            "Topic :: Utilities",
            "License :: OSI Approved :: GNU General Public License (GPL)",
        ],

        zip_safe=False
    )


def get_py_modules():
    py_modules=[]
    for file in glob.glob('*.py'):
        py_modules.append(os.path.splitext(file)[0])

    print(py_modules)
    return py_modules


if __name__ == "__main__":
    main()
