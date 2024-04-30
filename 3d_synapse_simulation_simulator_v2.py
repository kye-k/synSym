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

#%%

# DOUBLE CHECK SIGMA CALCULATION
# DOUBLE CHECK DIFFUSION CALCULATION
# ADD NEW COLUMN TO SPECIFY WHETHER IN AXON OR SYNAPSE TO EACH POINT
# NEED TO CAP BROWNIAN MOTION AT LIKE 3 standard deviations or something like that 

def bounce_particle(ray_origin, ray_direction, trimesh):
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
    locations, index_array, index_tri = trimesh.ray.intersects_location(ray_origins=np.array([ray_origin]), ray_directions=np.array([ray_direction]))
    
    # print('index_tri:')
    # print(index_tri)
    # print('locations:')
    # print(locations)
    # print('ray_origin:')
    # print(ray_origin)
    # print('ray_direction:')
    # print(ray_direction)
    
    # find the normal of the intercepted triangle
    plane_normal = trimesh.face_normals[index_tri][0]
    
    # getting the intercept
    intercept = locations[0]
    
    # calculating bounce
    d  = - np.dot(intercept, plane_normal)
    K = (np.dot(-plane_normal, ray_direction) - d) / np.dot(plane_normal, plane_normal)
    
    mid_point = plane_normal * K + ray_direction
    ref_point = mid_point * 2 - ray_direction
    
    return ref_point

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
                if not trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                    x_temp, y_temp, z_temp = bounce_particle([x, y, z], [x_temp, y_temp, z_temp], trimesh_axon)

            #if the particle is in the axon
            else:
                
                in_synapse = False
                
                x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
                
                # if exists axon from axon
                if not trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                    print('notinsynapse@step{}'.format(i))
                    print(repr(x_data), repr(y_data), repr(z_data))
                    print(x_temp, y_temp, z_temp)
                    print('bounce@step{}'.format(i))
                    x_temp, y_temp, z_temp = bounce_particle([x, y ,z], [x_temp, y_temp, z_temp], trimesh_axon)
                    
                        
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

# trimesh_axon = trimesh.load_mesh('bloss18_dataset_1_series_synapse_repaired.stl')
# trimesh_synapse = trimesh.load_mesh('bloss18_dataset_1_series_synapse_sv_repaired_hull.stl')

# # a check to fill the holes
# trimesh_axon.fill_holes()
# trimesh_synapse.fill_holes()

trimesh_axon = trimesh.creation.torus(major_radius=5, minor_radius=0.2)
trimesh_synapse = trimesh.creation.icosphere(radius=0.5)

D_synapse = 0.37
D_axon = 5
tau_sim = 0.0001
steps = 100
num_p_synapse = 0
num_p_axon = 1

# savefile = 'data\\simulation_5'
np.random.seed(2)
data, dict_store = simulate_particle_confined_3D_synapse_slowed(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse)

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

# mesh = trimesh.creation.icosphere()
# p0 = np.array([[3.945821329428025, 4.08235744597003, 1.7878529667085277], [3.9052941431285633, 4.024464524922043, 1.8156631345505077]])

# fig = go.Figure()

# fig.add_trace(go.Mesh3d(x=trimesh_axon.vertices[:, 0],
#                         y=trimesh_axon.vertices[:, 1],
#                         z=trimesh_axon.vertices[:, 2],
#                         i=trimesh_axon.faces[:, 0],
#                         j=trimesh_axon.faces[:, 1],
#                         k=trimesh_axon.faces[:, 2],
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='red'))

# fig.add_trace(go.Scatter3d(x=p0[:,0],
#                            y=p0[:,1],
#                            z=p0[:,2]))

# pio.write_html(fig, file='temp.html')

# #%%

# a = [-0.706223094559216 -5.103290006033867 -0.007194383559370975]
# b = [-0.7098916711313595 -5.125477421016757 -0.05113086135575021]
# c = [-0.7098082717420496 -5.124630652806871 -0.05104746196644032]

# p0 = np.array([a, b, bounce_particle(a, b, trimesh_axon)])

# fig = go.Figure()

# fig.add_trace(go.Mesh3d(x=trimesh_axon.vertices[:, 0],
#                         y=trimesh_axon.vertices[:, 1],
#                         z=trimesh_axon.vertices[:, 2],
#                         i=trimesh_axon.faces[:, 0],
#                         j=trimesh_axon.faces[:, 1],
#                         k=trimesh_axon.faces[:, 2],
#                         alphahull=-1,
#                         opacity=0.2,
#                         color='black'))


# # ray_origin = [-0.6656530177231605, -4.93036718898993, 0.019864991151483657]
# # ray_direction = [-0.9419615558267933, -5.154786150316589, -0.13208596548394003]
# # locations, index_array, index_tri = trimesh_axon.ray.intersects_location(ray_origins=np.array([ray_origin]), ray_directions=np.array([ray_direction]))


# # fig.add_trace(go.Scatter3d(x=p0[:,0],
# #                            y=p0[:,1],
# #                            z=p0[:,2]))

# # trig = trimesh_axon.triangles[index_tri]

# # fig.add_trace(go.Scatter3d(x=trig[:,0],
# #                            y=trig[:,1],
# #                            z=trig[:,2]))

# pio.write_html(fig, file='temp.html')


# ray_origins = p0[0]
# ray_directions = p0[1]

# #%% is the collision off??

# ray_origin = [-0.6656530177231605, -4.93036718898993, 0.019864991151483657]
# ray_direction = [-0.9419615558267933, -5.154786150316589, -0.13208596548394003]
# locations, index_array, index_tri = trimesh_axon.ray.intersects_location(ray_origins=np.array([ray_origin]), ray_directions=np.array([ray_direction]))

# # says its colliding with "1990"

#%% trouble shooting boucning code 20240308


# first find the problem (I think it is the intersects_location function); but lets double check...

trimesh_axon = trimesh.creation.icosphere(radius=3)
trimesh_synapse = trimesh.creation.icosphere(radius=0.5)

D_synapse = 0.37
D_axon = 6
tau_sim = 0.001
steps = 100
num_p_synapse = 0
num_p_axon = 1

np.random.seed(5)
data, dict_store = simulate_particle_confined_3D_synapse_slowed(D_axon, D_synapse, tau_sim, steps, trimesh_axon, trimesh_synapse, num_p_axon, num_p_synapse)

#%%
x=np.array([2.51166545, 2.5236726 , 2.45886206, 2.32820124, 2.39430824,
       2.52043681, 2.59107631, 2.49556673, 2.57360857, 2.57396885,
       2.50478364, 2.49905357, 2.64527891, 2.61177439, 2.65071072,
       2.77551949, 2.76449774, 2.62090974, 2.74712745, 2.81905651,
       2.91814747])

y = np.array([-0.06953287,  0.10381925,  0.12437016,  0.10192706, -0.08044139,
        0.12301903,  0.01559882, -0.0306846 , -0.02420568, -0.03580978,
       -0.03648839, -0.00918797, -0.01870471, -0.07103757, -0.04150301,
        0.01266022,  0.16749022,  0.09195213,  0.08021291,  0.09388481,
        0.06685917])

z = np.array([0.67046318, 0.57086176, 0.53472631, 0.49541857, 0.4187178 ,
       0.25317638, 0.15931282, 0.26846733, 0.22866862, 0.31554326,
       0.30447185, 0.32612443, 0.49718172, 0.50821704, 0.64974454,
       0.61290075, 0.63713792, 0.57387452, 0.82145681, 0.77372783,
       0.68118735])


p0 = np.stack((x,y,z), axis=1)
locations, index_array, index_tri = trimesh_axon.ray.intersects_location(ray_origins=np.array([p0[-2]]), ray_directions=np.array([p0[-1]]))

# figure
fig = go.Figure()

fig.add_trace(go.Scatter3d(x=x[-2:],
                           y=y[-2:],
                           z=z[-2:]))

fig.add_trace(go.Scatter3d(x=[locations[0][0]],
                           y=[locations[0][1]],
                           z=[locations[0][2]]))

fig.add_trace(go.Mesh3d(x=trimesh_axon.vertices[:, 0],
                        y=trimesh_axon.vertices[:, 1],
                        z=trimesh_axon.vertices[:, 2],
                        i=trimesh_axon.faces[:, 0],
                        j=trimesh_axon.faces[:, 1],
                        k=trimesh_axon.faces[:, 2],
                        alphahull=-1,
                        opacity=0.2,
                        color='black'))

fig.update_scenes(aspectmode='data')
pio.write_html(fig, file='troubleshoot.html')


#%% need to probably simplify it even more...




