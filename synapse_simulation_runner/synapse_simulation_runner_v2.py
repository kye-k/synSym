import numpy as np
import os
import pandas as pd
import trimesh
import time
import json
import copy

# unused imports
# from scipy.optimize import linprog
# from scipy.spatial import Delaunay
# import plotly.express as px
# import plotly.io as pio
# import concurrent.futures
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

class particle():
    def __init__(self, tr_id, initial_location):
        self.tr_id = tr_id
        self.initial_location = initial_location

class simulation_parameters():
    def __init__(self):
        self.exists = True

    def read_parameter_json(self, fname):

        with open(fname) as f:
            parameter_dict = json.load(f)

        # initiate variables from parameter_dict
        self.simulation_type = parameter_dict['simulation_type']
        self.num_p_total = parameter_dict['num_p_total']
        self.num_p_synapse = parameter_dict['num_p_synapse']
        self.num_p_axon = parameter_dict['num_p_axon']
        self.D_synapse = parameter_dict['D_synapse']
        self.D_axon = parameter_dict['D_axon']
        self.trimesh_synapse_fname = parameter_dict['trimesh_synapse']
        self.trimesh_axon_fname = parameter_dict['trimesh_axon']
        self.tau_sim = parameter_dict['tau_sim']
        self.steps = parameter_dict['steps']
        self.savefile = parameter_dict['savefile']
        self.seed = parameter_dict['seed']
        self.leave_prob = parameter_dict['leave_prob']
        self.pON = parameter_dict['pON']
        self.pOFF = parameter_dict['pOFF']
        print('seed: {}'.format(self.seed))

        # load trimesh
        self.trimesh_axon = trimesh.load_mesh(self.trimesh_axon_fname)
        self.trimesh_synapse = trimesh.load_mesh(self.trimesh_synapse_fname)
        self.trimesh_synapse2 = trimesh.load_mesh(self.trimesh_synapse_fname)

        # check trimesh properties
        trimesh_axon_properties = {'is_water_tight': self.trimesh_axon.is_watertight,
                                   'is_volume': self.trimesh_axon.is_volume,
                                   'is_winding_consistent': self.trimesh_axon.is_winding_consistent,
                                   'is_convex': self.trimesh_axon.is_convex,
                                   'is_empty': self.trimesh_axon.is_empty}
        
        trimesh_synapse_properties = {'is_water_tight': self.trimesh_synapse.is_watertight,
                                      'is_volume': self.trimesh_synapse.is_volume,
                                      'is_winding_consistent': self.trimesh_synapse.is_winding_consistent,
                                      'is_convex': self.trimesh_synapse.is_convex,
                                      'is_empty': self.trimesh_synapse.is_empty}


        print('trimesh_axon: \n{}'.format(trimesh_axon_properties))
        print('trimesh_synapse: \n{}'.format(trimesh_synapse_properties))

        # fill holes of trimesh
        self.trimesh_axon.fill_holes()
        self.trimesh_synapse.fill_holes()

        # generating meshes to perform sampleing (initialization) with; as using original meshes spits error during multiprocess
        self.trimesh_axon_sampler = copy.deepcopy(self.trimesh_axon)
        self.trimesh_synapse_sampler = copy.deepcopy(self.trimesh_synapse)
    
    def initialize_particles(self):

        # if seed is given
        if self.seed != None:
            np.random.seed(self.seed)

        # if num_p_total is given, the simulation simply generates particles through the whole axon length
        if self.num_p_total != None:
            # initial positions for whole axon
            print("initializing particles through entire axon")
            total_p_array = np.array([], dtype=np.int64).reshape(0,3)
            while len(total_p_array) < self.num_p_total:
                total_p_array = np.concatenate((total_p_array, trimesh.sample.volume_mesh(self.trimesh_axon_sampler, self.num_p_total)))
            total_p_array = total_p_array[:self.num_p_total]
        
        # else generate given number of particles in synapse and in axon
        else:
            # initial positions for synapse
            print("initializing particles in synapse")
            syn_array = np.array([], dtype=np.int64).reshape(0,3)
            while len(syn_array) < self.num_p_synapse:
                syn_array = np.concatenate((syn_array, trimesh.sample.volume_mesh(self.trimesh_synapse_sampler, self.num_p_synapse)))
            syn_array = syn_array[:self.num_p_synapse]
            
            # initial positions for axon
            print("initializing particles in axon")
            axon_array = np.array([], dtype=np.int64).reshape(0,3)
            while len(axon_array) < self.num_p_axon:
                temporary_array = trimesh.sample.volume_mesh(self.trimesh_axon_sampler, self.num_p_axon) # find particles in axon
                index_array = np.invert(self.trimesh_synapse_sampler.contains(temporary_array)) # find those particles that are in synapse area and exclude
                axon_array = np.concatenate((axon_array, temporary_array[index_array]))
            axon_array = axon_array[:self.num_p_axon]

            # total array of particles
            total_p_array = np.concatenate((syn_array, axon_array))
        
        particle_list = []
        for tr_id in range(len(total_p_array)):
            particle_list.append(particle(tr_id, total_p_array[tr_id]))

        return particle_list

# molecular crowding (slowed motion) ~ DONE
def simulate_particle_mc(sp, particle):

    start_time = time.process_time()
    
    if sp.seed != None:
            np.random.seed(sp.seed)

    # calculate sigma
    s_axon = np.sqrt(2 * sp.D_axon * sp.tau_sim)
    s_synapse = np.sqrt(2 * sp.D_synapse * sp.tau_sim)
    
    # vectors for storage
    x_data = np.zeros(sp.steps + 1)
    y_data = np.zeros(sp.steps + 1)
    z_data = np.zeros(sp.steps + 1)
    t_data = np.arange(0, sp.steps + 1)
    in_synapse_data = np.zeros(sp.steps + 1)
        
    # initial location
    x = particle.initial_location[0]
    y = particle.initial_location[1]
    z = particle.initial_location[2]
        
    # store initial locations
    x_data[0] = x
    y_data[0] = y
    z_data[0] = z
    t_data[0] = 0

    # for each step
    for i in range(1, sp.steps + 1):
        
        # if the particle is in the synapse
        if sp.trimesh_synapse.contains([[x, y, z]]):
            
            in_synapse = True
            
            # brownian motion (create temporary location for check)
            x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
        
            # if exists axon from synapse
            while not sp.trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
        
        #if the particle is in the axon
        else:
            
            in_synapse = False
            
            x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
            
            # if exists axon from axon
            while not sp.trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]

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
    in_synapse = sp.trimesh_synapse.contains([[x, y, z]])[0]
    in_synapse_data[sp.steps] = in_synapse
    
    # create dataframe to save
    tr_df = pd.DataFrame({'tr': [particle.tr_id] * (sp.steps + 1),
                            'x': x_data,
                            'y': y_data,
                            'z': z_data,
                            't': t_data,
                            'in_synapse': in_synapse_data})
        
    # savefile for trajectory simulated
    write_start_time = time.time()
    if sp.savefile != None:
        tr_df.to_csv(os.path.join(sp.savefile, 'tr_id_{}.csv'.format(particle.tr_id)), index=False)

    return 'particle {} complete in {}s... write time was {}s'.format(particle.tr_id, time.process_time() - start_time, time.time() - write_start_time)

# interactions with spatially clustered binding sites (ICBS) ~ DONE
def simulate_particle_icbs(sp, particle):

    start_time = time.process_time()
    
    if sp.seed != None:
            np.random.seed(sp.seed)

    # calculate sigma
    s_axon = np.sqrt(2 * sp.D_axon * sp.tau_sim)
    s_synapse = np.sqrt(2 * sp.D_synapse * sp.tau_sim)
    
    # vectors for storage
    x_data = np.zeros(sp.steps + 1)
    y_data = np.zeros(sp.steps + 1)
    z_data = np.zeros(sp.steps + 1)
    t_data = np.arange(0, sp.steps + 1)
    in_synapse_data = np.zeros(sp.steps + 1)
        
    # initial location
    x = particle.initial_location[0]
    y = particle.initial_location[1]
    z = particle.initial_location[2]
        
    # store initial locations
    x_data[0] = x
    y_data[0] = y
    z_data[0] = z
    t_data[0] = 0
    is_bound = False

    # store initial location into temps
    x_temp = x
    y_temp = y
    z_temp = z

    # for each step
    for i in range(1, sp.steps + 1):
        
        # if the particle is in the synapse
        if sp.trimesh_synapse.contains([[x, y, z]]):
            
            in_synapse = True
            
            # if the particle is bound
            if is_bound:

                # x_temp, y_temp, z_temp remain constant; not updating these values
                is_bound = np.random.uniform(0, 1) >= sp.pOFF # check if unbinds at end of step
            
            # else the particle is not bound
            else:
                # brownian motion (create temporary location for check)
                x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]

                # if exits axon from synapse
                while not sp.trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                    x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
                
                is_bound = np.random.uniform(0, 1) < sp.pON # check if binds at end of step
        
        #if the particle is in the axon
        else:
            
            in_synapse = False
            
            x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
            
            # if exists axon from axon
            while not sp.trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]

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
    in_synapse = sp.trimesh_synapse.contains([[x, y, z]])[0]
    in_synapse_data[sp.steps] = in_synapse
    
    # create dataframe to save
    tr_df = pd.DataFrame({'tr': [particle.tr_id] * (sp.steps + 1),
                            'x': x_data,
                            'y': y_data,
                            'z': z_data,
                            't': t_data,
                            'in_synapse': in_synapse_data})
        
    # savefile for trajectory simulated
    if sp.savefile != None:
        tr_df.to_csv(os.path.join(sp.savefile, 'tr_id_{}.csv'.format(particle.tr_id)), index=False)

    return 'particle {} complete in {}s'.format(particle.tr_id, time.process_time() - start_time)


# confining boundary (CB)
def simulate_particle_cb(sp, particle): # needs sp.leave_prob

    start_time = time.process_time()
    start_time_clock = time.time()

    if sp.seed != None:
        np.random.seed(sp.seed)

    # calculate sigma
    s_axon = np.sqrt(2 * sp.D_axon * sp.tau_sim)
    s_synapse = np.sqrt(2 * sp.D_synapse * sp.tau_sim)

    # vectors for storage
    x_data = np.zeros(sp.steps + 1)
    y_data = np.zeros(sp.steps + 1)
    z_data = np.zeros(sp.steps + 1)
    t_data = np.arange(0, sp.steps + 1)
    in_synapse_data = np.zeros(sp.steps + 1)
    
    # initial location
    x = particle.initial_location[0]
    y = particle.initial_location[1]
    z = particle.initial_location[2]

    # store initial locations
    x_data[0] = x
    y_data[0] = y
    z_data[0] = z
    t_data[0] = 0
    is_bound = False

    # store initial location into temps
    x_temp = x
    y_temp = y
    z_temp = z

    # for each step
    for i in range(1, sp.steps + 1):
        
        # if the particle is in the synapse
        if sp.trimesh_synapse.contains([[x, y, z]]):
            
            in_synapse = True
            
            # brownian motion (create temporary location for check)
            x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
            
            # determine if particle can leave in this step
            can_leave = np.random.uniform(0, 1) < sp.leave_prob
            
            # if it attempts an exit from the synapse, but can not leave
            if not can_leave:
                while not sp.trimesh_synapse.contains([[x_temp, y_temp, z_temp]]):
                    x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]
            
            # if exists axon from synapse
            while not sp.trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                x_temp, y_temp, z_temp = np.random.randn(3) * s_synapse + [x, y, z]                       
        
        #if the particle is in the axon
        else:
            
            in_synapse = False
            
            x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
            
            # if exists axon from axon
            while not sp.trimesh_axon.contains([[x_temp, y_temp, z_temp]]):
                x_temp, y_temp, z_temp = np.random.randn(3) * s_axon + [x, y, z]
        
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
    in_synapse = sp.trimesh_synapse.contains([[x, y, z]])[0]
    in_synapse_data[sp.steps] = in_synapse
    
    # create dataframe to save
    tr_df = pd.DataFrame({'tr': [particle.tr_id] * (sp.steps + 1),
                            'x': x_data,
                            'y': y_data,
                            'z': z_data,
                            't': t_data,
                            'in_synapse': in_synapse_data})
    
    # savefile for trajectory simulated
    write_start_time = time.time()
    if sp.savefile != None:
        tr_df.to_csv(os.path.join(sp.savefile, 'tr_id_{}.csv'.format(particle.tr_id)), index=False)

    return 'particle {} complete in {}s... write time was {}s... clock time was {}s'.format(particle.tr_id, time.process_time() - start_time, time.time() - write_start_time, time.time() - start_time_clock)
