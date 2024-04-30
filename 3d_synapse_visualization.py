# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:59:23 2024

@author: uqkkudo
"""

#%% import


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from scipy.optimize import linprog
from scipy.spatial import Delaunay
from scipy.constants import Avogadro
from stl import mesh as stlmesh
import plotly.express as px
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
import trimesh
import time
import seaborn as sns
import json
import glob
from PIL import Image
import re


os.chdir(os.path.dirname(__file__))
plt.rcParams['figure.dpi'] = 800
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.sans-serif'] = "Arial"

pio.renderers.default='browser'

#%% functions

def load_simulation_data(folder):
    
    dict_store = {}
    csv_list = os.listdir(folder)
    
    for csv in csv_list:
        
        if '.csv' in csv:
            tr_id = float(os.path.splitext(csv)[0].split('_')[-1])
            dict_store[tr_id] = pd.read_csv(os.path.join(folder, csv))
    
    data = pd.concat(dict_store)
    data = data.reset_index(drop=True)
    print('data successfully loaded')
    
    return data, dict_store

fname = 'data\\simulation_9'

data, dict_store = load_simulation_data(fname)

#%%

# load the meshes
trimesh_axon = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
trimesh_synapse = trimesh.load_mesh('bloss18_dataset_1_series_synapse_sv_repaired_hull.stl')

# a check to fill the holes
trimesh_axon.fill_holes()
trimesh_synapse.fill_holes()

plot_data = dict_store[300]
plot_data = plot_data.query('t < 1000')

fig = go.Figure(data=go.Scatter3d(
    x=plot_data.x, y=plot_data.y, z=plot_data.z,
    marker=dict(
        size=8,
        color=plot_data.t
    ),
    line=dict(
        color='black',
        width=4
    )
))

# # wrangling the mesh for plotting with plotly
# flattened_vectors = np.reshape(axon_mesh.vectors, newshape=(axon_mesh.vectors.shape[0] * axon_mesh.vectors.shape[1], 3))
# unique_vectors, indices = np.unique(flattened_vectors, axis=0, return_inverse=True)

# x = unique_vectors[:, 0]
# y = unique_vectors[:, 1]
# z = unique_vectors[:, 2]

# indices_stacked = np.reshape(indices, newshape=(axon_mesh.vectors.shape[0], axon_mesh.vectors.shape[1]))

# i = indices_stacked[:, 0]
# j = indices_stacked[:, 1]
# k = indices_stacked[:, 2]

# fig.add_trace(go.Mesh3d(x=x,
#                         y=y,
#                         z=z,
#                         i=i,
#                         j=j,
#                         k=k,
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='black'))

fig.add_trace(go.Mesh3d(x=trimesh_synapse.vertices[:, 0],
                        y=trimesh_synapse.vertices[:, 1],
                        z=trimesh_synapse.vertices[:, 2],
                        i=trimesh_synapse.faces[:, 0],
                        j=trimesh_synapse.faces[:, 1],
                        k=trimesh_synapse.faces[:, 2],
                        alphahull=-1,
                        opacity=0.2,
                        color='red'))

fig.add_trace(go.Mesh3d(x=trimesh_axon.vertices[:, 0],
                        y=trimesh_axon.vertices[:, 1],
                        z=trimesh_axon.vertices[:, 2],
                        i=trimesh_axon.faces[:, 0],
                        j=trimesh_axon.faces[:, 1],
                        k=trimesh_axon.faces[:, 2],
                        alphahull=-1,
                        opacity=0.2,
                        color='black'))

pio.write_html(fig, file='temp.html')


#%% DENSITY OVER TIME
# need to make vmax and vmin 
tau_sim = 0.0001

time_points_of_interest = np.round(np.linspace(0, 500, 50))

x_bounds = (trimesh_axon.vertices[:, 0].min() - 0.2, trimesh_axon.vertices[:, 0].max() + 0.2)
y_bounds = (trimesh_axon.vertices[:, 1].min() - 0.2, trimesh_axon.vertices[:, 1].max() + 0.2)
z_bounds = (trimesh_axon.vertices[:, 2].min() - 0.2, trimesh_axon.vertices[:, 2].max() + 0.2)

for time_point in time_points_of_interest:
    temp_df = data.query('t == @time_point')
    fig, ax = plt.subplots()
    sns.histplot(x=temp_df.z, y=temp_df.y, binwidth=0.02, ax=ax, vmin=0, vmax=10)
    ax.set_aspect('equal')
    ax.set_xlim(z_bounds)
    ax.set_ylim(y_bounds)
    ax.set_title(time_point * tau_sim)
    fig.savefig('plots/{}.png'.format(int(time_point)))
    plt.close()
    
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted_alphanumeric(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
                save_all=True, duration=300, loop=0)
    
make_gif('plots')
    
    
#%% PERMEABILITY
# find the rate at which molecules enter and exit

def find_fraction_in_synapse(data):
    df_in_synapse = data.groupby('t').sum().in_synapse.reset_index()
    return df_in_synapse

df_in_synapse = find_fraction_in_synapse(data)

fig, ax = plt.subplots()
sns.lineplot(df_in_synapse, x='t', y='in_synapse', ax=ax)
sns.despine(ax=ax)
ax.set_ylabel('Particles in Synapse')
ax.set_xlabel('Frame (t)')

    

### FRACTION AT EQUILIBRIUM
# at equilibrium what is the concentration of particles within and outside of the presynapse

avg_time_window = 1000 # time-window for equilibrium calculation; will use the last frames of the movie here
data.tr.max()
eqSynapse = df_in_synapse[-avg_time_window:].in_synapse.mean()
eqAxon = len(data.tr.unique()) - eqSynapse

C_synapse = (eqSynapse / Avogadro) / (trimesh_synapse.volume*1e-15)
C_axon = (eqAxon / Avogadro) / ((trimesh_axon.volume - trimesh_synapse.volume)*1e-15) # 1e-18 for m cubed; * 1000 for L


### MAKE SURE THAT THE VOLUME IS CORRECT CALCULATION
