# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 08:27:48 2024

@author: uqkkudo
"""
##%% imports

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
import json

os.chdir(os.path.dirname(__file__))
plt.rcParams['figure.dpi'] = 800
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.sans-serif'] = "Arial"

pio.renderers.default='browser'

##%%

def simulate_particle_confined_3D_synapse_slowed(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse, **kwargs):
    
    dict_store = {}
 
    savefile = kwargs.get('savefile', None)
    seed = kwargs.get('seed', None)
    
    if seed != None:
        np.random.seed(seed)
    
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

def simulate_particle_confined_3D_synapse_boundary(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse, leave_prob, **kwargs):
    
    dict_store = {}
 
    savefile = kwargs.get('savefile', None)
    seed = kwargs.get('seed', None)
    
    if seed != None:
        np.random.seed(seed)
    
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
                
                # determine if particle can leave in this step
                can_leave = np.random.uniform(0, 1) < leave_prob
                
                # if it attempts and exit from the synapse, but can not leave
                if not can_leave:
                    while not trimesh_synapse.contains([[x_temp, y_temp, z_temp]]):
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

def simulate_particle_confined_3D_synapse_sparseHA(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse, pON, pOFF, **kwargs):
    
    dict_store = {}
 
    savefile = kwargs.get('savefile', None)
    seed = kwargs.get('seed', None)
    
    if seed != None:
        np.random.seed(seed)
    
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
        is_bound = False

        # for each step
        for i in range(1, steps + 1):
            
            # if the particle is in the synapse
            if trimesh_synapse.contains([[x, y, z]]):
                
                in_synapse = True
                
                # if the particle is bound
                if is_bound:
                    
                    # x_temp, y_temp, z_temp remain constant
                    is_bound = np.random.uniform(0, 1) >= pOFF # check if unbinds at end of step
                    
                # if the particle is not bound
                else:
                    # brownian motion (create temporary location for check)
                    x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
            
                    # if exists axon from synapse
                    while not trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                        x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
                    
                    is_bound = np.random.uniform(0, 1) < pON # check if binds at end of step
            
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
        
        if '.csv' in csv:
            tr_id = float(os.path.splitext(csv)[0].split('_')[-1])
            dict_store[tr_id] = pd.read_csv(os.path.join(folder, csv))
    
    data = pd.concat(dict_store)
    data = data.reset_index(drop=True)
    
    return data, dict_store

##%% RUN SIMULATION (1)

# # load the meshes
# trimesh_axon = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
# trimesh_synapse = trimesh.load_mesh('bloss18_dataset_1_series_synapse_sv_repaired_hull.stl')

# # a check to fill the holes
# trimesh_axon.fill_holes()
# trimesh_synapse.fill_holes()

# np.random.seed(1)

# D_synapse = 0.37
# D_axon = 3
# tau_sim = 0.0001
# steps = 10000
# num_p_synapse = 0
# num_p_axon = 1

# savefile = 'data\\simulation_6'

# data, dict_store = simulate_particle_confined_3D_synapse_slowed(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse)

# #%% RUN SIMULATION (2)

# # load the meshes
# trimesh_axon = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
# trimesh_synapse = trimesh.load_mesh('bloss18_dataset_1_series_synapse_sv_repaired_hull.stl')

# # a check to fill the holes
# trimesh_axon.fill_holes()
# trimesh_synapse.fill_holes()

# D_synapse = 0.5
# D_axon = 0.5
# tau_sim = 0.0001
# steps = 10000
# num_p_synapse = 1
# num_p_axon = 0
# pON = 0.5
# pOFF = 0.01

# savefile = 'data\\simulation_7'

# data, dict_store = simulate_particle_confined_3D_synapse_boundary(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse, leave_prob=leave_prob, savefile=savefile)


# #%% READ DATA

# # data, dict_store = load_simulation_data('data\\simulation_5')

# #%% plotting both meshes


# #%%

def read_parameter_json(fname):
    with open(fname) as f:
        parameter_dictionary = json.load(f)
    return parameter_dictionary

def create_parameter_json(fname, parameter_dictionary):
    with open(fname, 'w') as f:
        json.dump(parameter_dictionary, f, indent="")

# # simulation type can be 'slowed', 'boundary' or 'sparseHA'
# fname = 'test.json'
# parameter_dictionary = {'simulation_type': 'slowed',
#                         'trimesh_axon': 'bloss18_dataset_1_series_synapse_repaired.stl',
#                         'trimesh_synapse': 'bloss18_dataset_1_series_synapse_sv_repaired_hull.stl',
#                         'num_p_synapse': 0,
#                         'num_p_axon': 1,
#                         'D_synapse': 0.37,
#                         'D_axon': 3,
#                         'tau_sim': 0.0001,
#                         'steps': 10000, 
#                         'seed': None,
#                         'savefile': None}

# create_parameter_json(fname, parameter_dictionary)

##%%

def main():
    
    fname = input('input file directory/name of json: ')
    para_dict = read_parameter_json(fname)
    
    trimesh_axon = trimesh.load_mesh(para_dict['trimesh_axon'])
    trimesh_synapse = trimesh.load_mesh(para_dict['trimesh_synapse'])

    # fill holes of trimesh
    trimesh_axon.fill_holes()
    trimesh_synapse.fill_holes()

    if para_dict['simulation_type'] == 'slowed':

        simulate_particle_confined_3D_synapse_slowed(D_axon=para_dict['D_axon'],
                                                     D_synapse=para_dict['D_synapse'],
                                                     tau_sim=para_dict['tau_sim'],
                                                     steps=para_dict['steps'],
                                                     trimesh_axon=trimesh_axon,
                                                     trimesh_synapse=trimesh_synapse,
                                                     num_p_axon=para_dict['num_p_axon'],
                                                     num_p_synapse=para_dict['num_p_synapse'],
                                                     seed=para_dict['seed'],
                                                     savefile=para_dict['savefile'])
    
    elif para_dict['simulation_type'] == 'boundary':
        
        simulate_particle_confined_3D_synapse_boundary(D_axon=para_dict['D_axon'],
                                                       D_synapse=para_dict['D_synapse'],
                                                       tau_sim=para_dict['tau_sim'],
                                                       steps=para_dict['steps'],
                                                       trimesh_axon=trimesh_axon,
                                                       trimesh_synapse=trimesh_synapse,
                                                       num_p_axon=para_dict['num_p_axon'],
                                                       num_p_synapse=para_dict['num_p_synapse'],
                                                       leave_prob=para_dict['leave_prob'],
                                                       seed=para_dict['seed'],
                                                       savefile=para_dict['savefile'])
        
    elif para_dict['simulation_type'] == 'sparseHA':
        
        simulate_particle_confined_3D_synapse_sparseHA(D_axon=para_dict['D_axon'],
                                                       D_synapse=para_dict['D_synapse'],
                                                       tau_sim=para_dict['tau_sim'],
                                                       steps=para_dict['steps'],
                                                       trimesh_axon=trimesh_axon,
                                                       trimesh_synapse=trimesh_synapse,
                                                       num_p_axon=para_dict['num_p_axon'],
                                                       num_p_synapse=para_dict['num_p_synapse'],
                                                       pON=para_dict['pON'],
                                                       pOFF=para_dict['pOFF'],
                                                       seed=para_dict['seed'],
                                                       savefile=para_dict['savefile'])
        
    else:
        print('no simulation type found')

main()


##%%
# fname = 'data\\simulation_8'

# data, dict_store = load_simulation_data(fname)

# # load the meshes
# trimesh_axon = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
# trimesh_synapse = trimesh.load_mesh('bloss18_dataset_1_series_synapse_sv_repaired_hull.stl')

# # a check to fill the holes
# trimesh_axon.fill_holes()
# trimesh_synapse.fill_holes()

# plot_data = dict_store[0]

# fig = go.Figure(data=go.Scatter3d(
#     x=plot_data.x, y=plot_data.y, z=plot_data.z,
#     marker=dict(
#         size=8,
#         color=plot_data.in_synapse
#     ),
#     line=dict(
#         color='black',
#         width=4
#     )
# ))

# # # wrangling the mesh for plotting with plotly
# # flattened_vectors = np.reshape(axon_mesh.vectors, newshape=(axon_mesh.vectors.shape[0] * axon_mesh.vectors.shape[1], 3))
# # unique_vectors, indices = np.unique(flattened_vectors, axis=0, return_inverse=True)

# # x = unique_vectors[:, 0]
# # y = unique_vectors[:, 1]
# # z = unique_vectors[:, 2]

# # indices_stacked = np.reshape(indices, newshape=(axon_mesh.vectors.shape[0], axon_mesh.vectors.shape[1]))

# # i = indices_stacked[:, 0]
# # j = indices_stacked[:, 1]
# # k = indices_stacked[:, 2]

# # fig.add_trace(go.Mesh3d(x=x,
# #                         y=y,
# #                         z=z,
# #                         i=i,
# #                         j=j,
# #                         k=k,
# #                         alphahull=-1,
# #                         opacity=0.2,
# #                         color='black'))

# fig.add_trace(go.Mesh3d(x=trimesh_synapse.vertices[:, 0],
#                         y=trimesh_synapse.vertices[:, 1],
#                         z=trimesh_synapse.vertices[:, 2],
#                         i=trimesh_synapse.faces[:, 0],
#                         j=trimesh_synapse.faces[:, 1],
#                         k=trimesh_synapse.faces[:, 2],
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='red'))

# fig.add_trace(go.Mesh3d(x=trimesh_axon.vertices[:, 0],
#                         y=trimesh_axon.vertices[:, 1],
#                         z=trimesh_axon.vertices[:, 2],
#                         i=trimesh_axon.faces[:, 0],
#                         j=trimesh_axon.faces[:, 1],
#                         k=trimesh_axon.faces[:, 2],
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='black'))

# pio.write_html(fig, file='temp.html')

