# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 05:55:20 2024

@author: uqkkudo
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from scipy.optimize import linprog
from scipy.spatial import Delaunay
from stl import mesh as stlmesh
import plotly.express as px
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
import trimesh
import time
import seaborn as sns

os.chdir(os.path.dirname(__file__))
plt.rcParams['figure.dpi'] = 800
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.sans-serif'] = "Arial"

pio.renderers.default='browser'

#%% importing mesh

axon_mesh = stlmesh.Mesh.from_file('bloss18_dataset_1_series_synapse.stl')
vertices = np.unique(axon_mesh.v0, axis=0)

#%% plotting matplotlib

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)

#%% simulation function

# def simulate_particle_confined_3D(D, tau_sim, steps, vertices, num_traj, **kwargs):
    
#     dict_store = {}
    
#     delaunay_obj = Delaunay(vertices, furthest_site=True)
    
#     x0 = kwargs.get('x0', None)
#     y0 = kwargs.get('y0', None)
#     z0 = kwargs.get('z0', None)    
    
#     # calculate sigma
#     s = np.sqrt(2 * D * tau_sim)
    
#     for tr_id in range(num_traj):
        
#         # specified initial location of trajectory
#         if x0 != None and y0 != None and z0 != None:
#             x = x0
#             y = y0
#             z = z0
        
#         # # otherwise choose random point in hull
#         # else:
            
#         #     # find boundary of hull
#         #     minx, miny, maxx, maxy = hull.bounds
            
#         #     # wrap until find a start location within hull
#         #     x = random.uniform(minx, maxx)
#         #     y = random.uniform(miny, maxy)
#         #     loc = Point((x, y))
#         #     while not hull.contains(loc):
#         #        x = random.uniform(minx, maxx)
#         #        y = random.uniform(miny, maxy)
#         #        loc = Point((x, y))
        
#         # vectors for storage
#         x_data = np.zeros(steps)
#         y_data = np.zeros(steps)
#         z_data = np.zeros(steps)
#         t_data = np.arange(0, steps)

#         # for each step
#         for i in range(steps):
            
#             # brownian motion (create temporary location for check)
#             x_temp = x + np.random.randn() * s
#             y_temp = y + np.random.randn() * s
#             z_temp = z + np.random.randn() * s
            
#             # simulate until point lays within hull
#             while not delaunay_obj.find_simplex([x_temp, y_temp, z_temp]) >= 0:
                
#                 x_temp = x + np.random.randn() * s
#                 y_temp = y + np.random.randn() * s
#                 z_temp = z + np.random.randn() * s
#                 print('re_attempt')
            
#             # store as new coordinates
#             x = x_temp
#             y = y_temp
#             z = z_temp
#             print('step_complete')
            
#             # store to vector
#             x_data[i] = x
#             y_data[i] = y
#             z_data[i] = z
    
#         dict_store[tr_id] = pd.DataFrame({'tr': [tr_id] * steps,
#                                           'x': x_data,
#                                           'y': y_data,
#                                           'z': z_data,
#                                           't': t_data})
        
#         print('particle {} complete'.format(tr_id))
    
#     data = pd.concat(dict_store)
#     data = data.reset_index(drop=True)

#     return data

#%% https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.contains

# DOUBLE CHECK SIGMA CALCULATION
# DOUBLE CHECK DIFFUSION CALCULATION

# lets attempt using TRIMESH...

axon_mesh = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
axon_mesh.fill_holes()

def simulate_particle_confined_3D(D, tau_sim, steps, trimesh, num_traj, **kwargs):
    
    dict_store = {}
    
    x0 = kwargs.get('x0', None)
    y0 = kwargs.get('y0', None)
    z0 = kwargs.get('z0', None)    
    
    # calculate sigma
    s = np.sqrt(2 * D * tau_sim)
    
    for tr_id in range(num_traj):
        
        # specified initial location of trajectory
        if x0 != None and y0 != None and z0 != None:
            x = x0
            y = y0
            z = z0
        
        # vectors for storage
        x_data = np.zeros(steps)
        y_data = np.zeros(steps)
        z_data = np.zeros(steps)
        t_data = np.arange(0, steps)

        # for each step
        for i in range(steps):
            
            # brownian motion (create temporary location for check)
            x_temp = x + np.random.randn() * s
            y_temp = y + np.random.randn() * s
            z_temp = z + np.random.randn() * s
            
            # simulate until point lays within hull
            while not trimesh.contains([[x_temp, y_temp, z_temp]]):
                
                x_temp = x + np.random.randn() * s
                y_temp = y + np.random.randn() * s
                z_temp = z + np.random.randn() * s
                # print('re_attempt')
            
            # store as new coordinates
            x = x_temp
            y = y_temp
            z = z_temp
            # print('step_complete')
            
            # store to vector
            x_data[i] = x
            y_data[i] = y
            z_data[i] = z
    
        dict_store[tr_id] = pd.DataFrame({'tr': [tr_id] * steps,
                                          'x': x_data,
                                          'y': y_data,
                                          'z': z_data,
                                          't': t_data})
        
        # print('particle {} complete'.format(tr_id))
    
    data = pd.concat(dict_store)
    data = data.reset_index(drop=True)

    return data


#%% testing simulation

D = 1
tau_sim = 0.0001
steps = 100
num_traj = 1
x0 = 4
y0 = 3.8
z0 = 3

data = simulate_particle_confined_3D(D, tau_sim, steps, axon_mesh, num_traj, x0=x0, y0=y0, z0=z0)

#%%

# wrangling the mesh (stlmesh) for plotting with plotly
# flattened_vectors = np.reshape(axon_mesh.vectors, newshape=(axon_mesh.vectors.shape[0] * axon_mesh.vectors.shape[1], 3))
# unique_vectors, indices = np.unique(flattened_vectors, axis=0, return_inverse=True)

# x = unique_vectors[:, 0]
# y = unique_vectors[:, 1]
# z = unique_vectors[:, 2]

# indices_stacked = np.reshape(indices, newshape=(axon_mesh.vectors.shape[0], axon_mesh.vectors.shape[1]))

# i = indices_stacked[:, 0]
# j = indices_stacked[:, 1]
# k = indices_stacked[:, 2]

# creating figure and plotting
fig = go.Figure(data=go.Scatter3d(
    x=data.x, y=data.y, z=data.z,
    marker=dict(
        size=8,
        color=data.t,
        colorscale='Viridis',
    ),
    line=dict(
        color='black',
        width=4
    )
))


fig.add_trace(go.Mesh3d(x=axon_mesh.vertices[:, 0],
                        y=axon_mesh.vertices[:, 1],
                        z=axon_mesh.vertices[:, 2],
                        i=axon_mesh.faces[:, 0],
                        j=axon_mesh.faces[:, 1],
                        k=axon_mesh.faces[:, 2],
                        alphahull=-1,
                        opacity=0.2,
                        color='black'))

# fig.add_trace(go.Mesh3d(x=x,
#                         y=y,
#                         z=z,
#                         i=i,
#                         j=j,
#                         k=k,
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='black'))

# figure out how to add the IJK things....

# #%%

# fig = go.Figure(data=go.Scatter3d(
#     x=data.x, y=data.y, z=data.z,
#     marker=dict(
#         size=4,
#         color=data.t,
#         colorscale='Viridis',
#     ),
#     line=dict(
#         color='black',
#         width=2
#     )
# ))

# fig.add_trace(go.Scatter3d(x=axon_mesh.v0[0],
#                            y=axon_mesh.v0[1],
#                            z=axon_mesh.v0[2],
#                            marker=dict(size=4, color='black')))

# fig.show()

#%%

# def FindIndex(vector):
#     seen = []
#     res = []
    
#     for i, n in enumerate(vector):
#         if tuple(n) not in seen:
#             res.append(i)
#             seen.append(n)
#         else:
#             res.append(res[np.where(seen == n)])
            
#     return seen, res


# FindIndex(axon_mesh.v0)


# https://stackoverflow.com/questions/56830995/find-the-indexes-of-unique-elements-of-a-list-in-python

#%% simulating with a synaptic region

# DOUBLE CHECK SIGMA CALCULATION
# DOUBLE CHECK DIFFUSION CALCULATION
# ADD NEW COLUMN TO SPECIFY WHETHER IN AXON OR SYNAPSE TO EACH POINT

trimesh_axon = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
trimesh_synapse = trimesh.load_mesh('bloss18_dataset_1_series_synapse_sv_repaired_hull.stl')

# a check to fill the holes
trimesh_axon.fill_holes()
trimesh_synapse.fill_holes()

def simulate_particle_confined_3D_synapse_slowed(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse, **kwargs):
    
    dict_store = {}
 
    savefile = kwargs.get('savefile', None)
    
    # calculate sigma
    s_axon = np.sqrt(2 * D_axon * tau_sim)
    s_synapse = np.sqrt(2 * D_synapse * tau_sim)
    
    # initial positions for synapse
    print("Initializing particles in synapse")
    syn_array = np.array([], dtype=np.int64).reshape(0,3)
    
    while len(syn_array) < num_p_synapse:
        syn_array = np.concatenate((syn_array, trimesh.sample.volume_mesh(trimesh_synapse, num_p_synapse)))
        
    syn_array = syn_array[:num_p_synapse]
    
    # initial positions for axon
    print("Initializing particles in axon")
    axon_array = np.array([], dtype=np.int64).reshape(0,3)
    
    while len(axon_array) < num_p_axon:
        temporary_array = trimesh.sample.volume_mesh(trimesh_axon, num_p_axon) # find particles in axon
        index_array = np.invert(trimesh_synapse.contains(temporary_array)) # find those particles that are in synapse area and exclude
        axon_array = np.concatenate((axon_array, temporary_array[index_array]))
    
    axon_array = axon_array[:num_p_axon]
    
    total_p_array = np.concatenate((syn_array, axon_array))
    
    for tr_id in range(num_p_axon + num_p_synapse):
        
        c = 0
        # timer
        start_time = time.process_time()
        
        # vectors for storage
        x_data = np.zeros(steps + 1)
        y_data = np.zeros(steps + 1)
        z_data = np.zeros(steps + 1)
        t_data = np.arange(0, steps + 1)
        in_synapse_data = np.zeros(steps + 1)
        
        # initial location
        x = total_p_array[tr_id, 0]
        y = total_p_array[tr_id, 1]
        z = total_p_array[tr_id, 2]
        
        # store initial locations
        x_data[0] = x
        y_data[0] = y
        z_data[0] = z
        t_data[0] = 0

        # for each step
        for i in range(1, steps + 1):
            
            # if the particle is in the synapse
            if trimesh_synapse.contains([[x, y, z]]):
                
                in_synapse = True
                
                # brownian motion (create temporary location for check)
                x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
            
                # if exists axon from synapse
                while not trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                    x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
            
            #if the particle is in the axon
            else:
                
                in_synapse = False
                
                x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
                
                # if exists axon from axon
                while not trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                    x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
                    c += 1

            
            # store as new coordinates
            x = x_temp
            y = y_temp
            z = z_temp
            
            # store to vector
            x_data[i] = x
            y_data[i] = y
            z_data[i] = z
            in_synapse_data[i - 1] = in_synapse

        
        # add the last point into in_synapse_data
        in_synapse = trimesh_synapse.contains([[x, y, z]])[0]
        in_synapse_data[steps] = in_synapse
        
        # create dataframe to save
        tr_df = pd.DataFrame({'tr': [tr_id] * (steps + 1),
                              'x': x_data,
                              'y': y_data,
                              'z': z_data,
                              't': t_data,
                              'in_synapse': in_synapse_data})
        
        # store into dictionary
        dict_store[tr_id] = tr_df
        
        # savefile for trajectory simulated
        if savefile != None:
            tr_df.to_csv(os.path.join(savefile, 'tr_id_{}.csv'.format(tr_id)))
        
        print('particle {} complete in {}s'.format(tr_id, time.process_time() - start_time))
        print(c)
    
    data = pd.concat(dict_store)
    data = data.reset_index(drop=True)

    return data, dict_store

def load_simulation_data(folder):
    
    dict_store = {}
    csv_list = os.listdir(folder)
    
    for csv in csv_list:
        
        tr_id = float(os.path.splitext(csv)[0].split('_')[-1])
        dict_store[tr_id] = pd.read_csv(os.path.join(folder, csv))
    
    data = pd.concat(dict_store)
    data = data.reset_index(drop=True)
    
    return data, dict_store

#%% RUN SIMULATION

D_synapse = 0.37
D_axon = 3
tau_sim = 0.0001
steps = 100
num_p_synapse = 0
num_p_axon = 1

savefile = 'data\\simulation_5'

data, dict_store = simulate_particle_confined_3D_synapse_slowed(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse, savefile=savefile)

#%% READ DATA

data, dict_store = load_simulation_data('data\\simulation_5')

#%% plotting both meshes

plot_data = dict_store[0]

fig = go.Figure(data=go.Scatter3d(
    x=plot_data.x, y=plot_data.y, z=plot_data.z,
    marker=dict(
        size=8,
        color=plot_data.in_synapse
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
#%% test plot for random sampling

# sample = trimesh.sample.volume_mesh(mesh, 100)


# fig = go.Figure(data=go.Scatter3d(
#     x=sample[:, 0], y=sample[:, 1], z=sample[:, 2],
#     marker=dict(
#         size=8,
#         colorscale='Viridis')
#     ))

# fig.add_trace(go.Mesh3d(x=mesh.vertices[:, 0],
#                         y=mesh.vertices[:, 1],
#                         z=mesh.vertices[:, 2],
#                         i=mesh.faces[:, 0],
#                         j=mesh.faces[:, 1],
#                         k=mesh.faces[:, 2],
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='black'))

#%% plot of density over time

time_points_of_interest = np.linspace(0, 999, 10)

for time_point in time_points_of_interest:
    temp_df = data.query('t == @time_point')
    fig, ax = plt.subplots()
    sns.histplot(x=temp_df.z, y=temp_df.y, binwidth=0.02, ax=ax)
    ax.set_aspect('equal')
    




#%%

data_to_plot_1 = data.query('t == 0')
data_to_plot_2 = data.query('t == 9999')

fig, ax = plt.subplots(2, 1, sharey=True)
# sns.histplot(x=data_to_plot_1.z, y=data_to_plot_1.y, binwidth=0.02, ax=ax[0], cbar=True)
# sns.histplot(x=data_to_plot_2.z, y=data_to_plot_2.y, binwidth=0.02, ax=ax[1], cbar=True)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

# sns.scatterplot(x=data.query('t==99').z, y=data.query('t==99').y, color='black', alpha=1, s=5, ax=ax)



### PERMEABILITY
# find the rate at which molecules enter and exit

def find_fraction_in_synapse(data):
    df_in_synapse = data.groupby('t').sum().in_synapse.reset_index()
    return df_in_synapse

df_in_synapse = find_fraction_in_synapse(data)

fig, ax = plt.subplots()
sns.lineplot(df_in_synapse, x='t', y='in_synapse')
    


### FRACTION AT EQUILIBRIUM
# at equilibrium what is the concentration of particles within and outside of the presynapse


#%% attempting updated boundary code

mesh = trimesh.creation.icosphere()
p0 = np.array([[0.2, 0.2, 0.2], [1, 1, -0.5], [0.3444808638328112, 0.3783957521384713, -0.4270129677147054]])

fig = go.Figure()

fig.add_trace(go.Mesh3d(x=mesh.vertices[:, 0],
                        y=mesh.vertices[:, 1],
                        z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0],
                        j=mesh.faces[:, 1],
                        k=mesh.faces[:, 2],
                        alphahull=-1,
                        opacity=0.2,
                        color='red'))

fig.add_trace(go.Scatter3d(x=p0[:,0],
                           y=p0[:,1],
                           z=p0[:,2]))

pio.write_html(fig, file='temp.html')


ray_origins = p0[0]
ray_directions = p0[1]

# start = time.process_time()    

# for i in range(9999):
        
#     mesh.ray.intersects_locaiton(ray_origins=ray_origins, ray_directions=ray_directions)

# print(time.process_time() - start)

# start = time.process_time()

# for i in range(9999):
#     mesh.contains(ray_directions)

# print(time.process_time() - start)

# # okay; so 

# locations, index_array, index_tri = mesh.ray.intersects_location(ray_origins=np.array([ray_origin]), ray_directions=np.array([ray_direction]))
# plane_normal = mesh.face_normals[index_tri][0]
# intercept = locations[0]

def bounce_particle(ray_origin, ray_direction, mesh):
    '''

    Parameters
    ----------
    ray_origin : TYPE
        np.array with shape (3,)
    ray_direction : TYPE
        np.array with shape (3,).
    mesh : TYPE
        DESCRIPTION.
        trimesh of bouncing particle

    Returns
    -------
    ref_point : TYPE
        location of particle after bounce np.array with shape (3,)

    '''
    
    # find locations where ray intercepts mesh, and the index of triangle 
    locations, index_array, index_tri = mesh.ray.intersects_location(ray_origins=np.array([ray_origin]), ray_directions=np.array([ray_direction]))
    
    # find the normal of the intercepted triangle
    plane_normal = mesh.face_normals[index_tri][0]
    
    # getting the intercept
    intercept = locations[0]
    
    # calculating bounce
    d  = - np.dot(intercept, plane_normal)
    K = (np.dot(-plane_normal, ray_direction) - d) / np.dot(plane_normal, plane_normal)
    
    mid_point = plane_normal * K + ray_direction
    ref_point = mid_point * 2 - ray_direction
    
    return ref_point

bounce_particle(ray_origins, ray_directions, mesh).shape






#%%


#%%
