"""
This script is run in threedsimhpc.
This script is used on the terminal (use an interactive job on HPC).
This script takes user input to generate figures and / or perform data analysis on data on the HPC.

@author: uqkkudo
"""

### IMPORTING

import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
import glob
from scipy import stats
from plotly.subplots import make_subplots
import json
from io import StringIO
from synapse_simulation_runner.synapse_simulation_runner_v2 import simulation_parameters
import seaborn as sns

os.chdir(os.path.dirname(__file__))
plt.rcParams['figure.dpi'] = 800
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.size': 10})
# plt.rcParams['font.sans-serif'] = "Arial"
pio.renderers.default='browser'

### FUNCTIONS

def load_simulation_data_json(fname):
    '''
    loads the simulation data given a directory

    '''
    print('loading {}'.format(fname))

    with open(fname, 'r') as output:
        dict_store = json.load(output)

    for key, item in dict_store.items():
        dict_store[key] = pd.read_json(StringIO(item))

    data = pd.concat(dict_store)
    data = data.reset_index(drop=True)
    print('data successfully loaded')

    return data, dict_store

def plot_example_trajectories(sp, dict_store, output_dir, ex_traj_number):
    '''
    creates example plots of trajectories

    '''

    for tr_id in np.random.choice(list(dict_store.keys()), ex_traj_number):
        plot_data = dict_store[tr_id]
        
        fig = go.Figure(data=go.Scatter3d(
            x=plot_data.x, y=plot_data.y, z=plot_data.z,
            marker=dict(
                size=4,
                color=plot_data.t,
                colorscale='deep'),
            line=dict(
                color=plot_data.in_synapse,
                colorscale='purp',
                width=2)))
        
        fig.add_trace(go.Mesh3d(x=sp.trimesh_synapse.vertices[:, 0],
                                y=sp.trimesh_synapse.vertices[:, 1],
                                z=sp.trimesh_synapse.vertices[:, 2],
                                i=sp.trimesh_synapse.faces[:, 0],
                                j=sp.trimesh_synapse.faces[:, 1],
                                k=sp.trimesh_synapse.faces[:, 2],
                                alphahull=-1,
                                opacity=0.2,
                                color='red'))
        
        fig.add_trace(go.Mesh3d(x=sp.trimesh_axon.vertices[:, 0],
                                y=sp.trimesh_axon.vertices[:, 1],
                                z=sp.trimesh_axon.vertices[:, 2],
                                i=sp.trimesh_axon.faces[:, 0],
                                j=sp.trimesh_axon.faces[:, 1],
                                k=sp.trimesh_axon.faces[:, 2],
                                alphahull=-1,
                                opacity=0.2,
                                color='black'))
        
        fig.update_scenes(aspectmode='data')
        pio.write_html(fig, file=os.path.join(output_dir, 'ex_tr_id_{}.html'.format(tr_id)))
        
def plot_density_over_time(sp, data, output_dir, sample_num=10):
    '''
    creates plot of density over time in plotly 

    '''

    time_points_of_interest = np.round(np.linspace(0, sp.steps, sample_num))

    fig = make_subplots(rows=1, cols=10,
                        specs=[[{'is_3d': True} for i in range(0, sample_num)]],
                        print_grid=False)

    # determining minimum and maximum kde values for colour palette
    plot_data = data.query('t == @time_points_of_interest[0]')
    pp = np.array([plot_data.x, plot_data.y, plot_data.z])
    kde = stats.gaussian_kde(pp, bw_method=0.5)
    kde_val = kde(pp)
    min_kde_val = min(kde_val)

    plot_data = data.query('t == @time_points_of_interest[@sample_num-1]')
    pp = np.array([plot_data.x, plot_data.y, plot_data.z])
    kde = stats.gaussian_kde(pp, bw_method=0.5)
    kde_val = kde(pp)
    max_kde_val = max(kde_val)

    for i in range(0, sample_num):
        
        # get the plot data
        plot_data = data.query('t == @time_points_of_interest[@i]')
        
        # calculate kde
        pp = np.array([plot_data.x, plot_data.y, plot_data.z])
        kde = stats.gaussian_kde(pp, bw_method=0.5)
        kde_val = kde(pp)
        
        fig.add_trace(go.Scatter3d(x=plot_data.x, y=plot_data.y, z=plot_data.z,
                                   mode='markers',
                                   marker=dict(
                                               size=4,
                                               color=kde_val,
                                               cmin=min_kde_val,
                                               cmax=max_kde_val)
                                   ),
                      row=1,
                      col=i+1
                      )
        
        fig.add_trace(go.Mesh3d(x=sp.trimesh_synapse.vertices[:, 0],
                                y=sp.trimesh_synapse.vertices[:, 1],
                                z=sp.trimesh_synapse.vertices[:, 2],
                                i=sp.trimesh_synapse.faces[:, 0],
                                j=sp.trimesh_synapse.faces[:, 1],
                                k=sp.trimesh_synapse.faces[:, 2],
                                alphahull=-1,
                                opacity=0.2,
                                color='red'),
                      row=1,
                      col=i+1)
        
        fig.add_trace(go.Mesh3d(x=sp.trimesh_axon.vertices[:, 0],
                                y=sp.trimesh_axon.vertices[:, 1],
                                z=sp.trimesh_axon.vertices[:, 2],
                                i=sp.trimesh_axon.faces[:, 0],
                                j=sp.trimesh_axon.faces[:, 1],
                                k=sp.trimesh_axon.faces[:, 2],
                                alphahull=-1,
                                opacity=0.2,
                                color='black'),
                      row=1,
                      col=i+1)

    fig.update_scenes(aspectmode='data')
    pio.write_html(fig, file=os.path.join(output_dir, 'density_plot.html'))

def find_fraction_in_synapse(data):
    df_in_synapse = data.groupby('t').sum().in_synapse.reset_index()
    return df_in_synapse

def generate_counts(df, col, val):
    return (df[col] == val).sum()

def plot_enrichment(sp, sp_inputdir, data, output_dir, avg_time_window):
    '''
    plots of enrichment and flux within the synaptic region

    '''

    # calculating equilibrium values
    df_in_synapse = find_fraction_in_synapse(data)
    eqSynapse = df_in_synapse[-avg_time_window:].in_synapse.mean()
    eqAxon = len(data.tr.unique()) - eqSynapse
    eqSynapse_density = eqSynapse / sp.trimesh_synapse.volume
    eqAxon_density = eqAxon / (sp.trimesh_axon.volume - sp.trimesh_synapse.volume)
    eqSynapse_enrichment = eqSynapse_density / eqAxon_density

    data['diff_in_synapse'] = data.in_synapse.diff()
    flux_exit = data.groupby('t').apply(generate_counts, 'diff_in_synapse', -1, include_groups=False) # -1 is EXITING
    flux_exit = flux_exit[1:] # disregarding the first step, as this is overlap of trajectories
    flux_entry = data.groupby('t').apply(generate_counts, 'diff_in_synapse', 1, include_groups=False) # +1 is ENTERING
    flux_entry = flux_entry[1:] # disregarding the first step, as this is overlap of trajectories

    eq_flux_exit = flux_exit[-avg_time_window:].mean()
    eq_flux_entry = flux_entry[-avg_time_window:].mean()

    # creating the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    sns.lineplot(df_in_synapse, x='t', y='in_synapse', ax=ax)
    sns.despine(ax=ax)
    ax.set_ylabel('Particles in Synapse')
    ax.set_xlabel('Frame (#)')
    ax.set_title('synapse enrichment: {}'.format(eqSynapse_enrichment))

    ax = axes[1]
    sns.lineplot(flux_exit, ax=ax)
    ax.set_title('flux_exit: {}'.format(eq_flux_exit))

    ax = axes[2]
    sns.lineplot(flux_entry, ax=ax)
    ax.set_title('flux_entry: {}'.format(eq_flux_entry))

    fig.savefig(os.path.join(output_dir, 'enrichment.png'))

    # exporting the data
    export_dict = {'avg_time_window': avg_time_window,
                   'eqSynapse': eqSynapse,
                   'eqAxon': eqAxon,
                   'eq_flux_exit': eq_flux_exit,
                   'eq_flux_entry': eq_flux_entry}
    
    for attr, value in sp.__dict__.items():
        if not any(i in attr for i in ['trimesh_axon', 'trimesh_synapse']):
            export_dict[attr] = value

    if os.path.exists(os.path.join(output_dir, 'export_data.json')):
        os.remove(os.path.join(output_dir, 'export_data.json'))

    with open(os.path.join(output_dir, 'export_data.json'), 'x') as f:
        json.dump(export_dict, f, indent="")

    df_in_synapse.to_csv(os.path.join(output_dir, 'df_in_synapse.csv'), index=False)

def main():

    print('please note that input simulation_parameter files must be in the form *sp.json, the axon and synapse .stl files must be in the directory')
    # main_input = input('what would you like to do:\n[1] create trajectory visualization\n[2] create density visualizations\n[3] create enrichment visualization\ninput: ')
    main_input = '1234'

    if any(i in main_input for i in ['1', '2', '3', '4']):

        # user inputs
        # path = input('please input directory containing *sp.jsons to visualize: ')
        path = 'example_simulation'
        # ex_traj_number = int(input('please input number of example trajectories you want to generate: '))
        ex_traj_number = 5
        # avg_time_window = int(input('please input the avg_time_window for enrichment calculation: '))
        avg_time_window = 500

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('sp.json'):

                    # specifying files
                    print('creating visualizations for: {}'.format(file))
                    spfname = os.path.join(root, file)

                    # create an output directory
                    output_dir = os.path.dirname(spfname) + '_vis'
                    if not os.path.isdir(output_dir):
                        os.mkdir(output_dir)
                    
                    print('checking if in {}'.format(os.path.join(output_dir, 'export_data.json')))

                    if os.path.exists(os.path.join(output_dir, 'export_data.json')):
                        print('passing')
                        continue

                    # reading simulation_parameters
                    sp = simulation_parameters()
                    sp.read_parameter_json(spfname)

                    # load data
                    dfname = os.path.join(root, 'output.json')
                    data, dict_store = load_simulation_data_json(dfname)
                    


                    if '1' in main_input:
                        plot_example_trajectories(sp, dict_store, output_dir, ex_traj_number)
                        print('plot_example_trajectories success')

                    if '2' in main_input:
                        plot_density_over_time(sp, data, output_dir)
                        print('plot_density_over_time success')

                    if '3' in main_input:
                        plot_enrichment(sp, spfname, data, output_dir, avg_time_window)
                        print('plot_enrichment success')
                    
                    plt.close('all')

    else:
        print('exiting')
        exit()

if __name__ == '__main__':
    main()
