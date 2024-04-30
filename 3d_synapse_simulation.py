# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:59:01 2024

@author: uqkkudo
"""
#%% import

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import copy
import tifffile
from cloudvolume import CloudVolume

os.chdir(os.path.dirname(__file__))

#%% loading data

w_size = 1000
z_size = 50

x_loc = 51642
y_loc = 59784
z_loc = 195

cv = CloudVolume("s3://open-neurodata/bloss/bloss18/image", mip=0, use_https=True)

# load data into numpy array
region = cv[x_loc-w_size:x_loc+w_size, y_loc-w_size:y_loc+w_size, z_loc-z_size:z_loc+z_size]

# save cutout as TIFF
tifffile.imwrite("data.tiff", data=np.transpose(region))