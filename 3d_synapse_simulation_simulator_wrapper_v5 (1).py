import os
import time
import json
import datetime
import concurrent.futures
import pandas as pd # type: ignore
from synapse_simulation_runner.synapse_simulation_runner_v2 import simulation_parameters
from synapse_simulation_runner.synapse_simulation_runner_v2 import simulate_particle_mc
from synapse_simulation_runner.synapse_simulation_runner_v2 import simulate_particle_icbs
from synapse_simulation_runner.synapse_simulation_runner_v2 import simulate_particle_cb

def main():
    
    # await user input
    # input_dir = input('input directory of where .jsons are located: ')
    # max_workers = int(input('number of processors to use: '))

    input_dir = 'simulation_ca1_cb_p_alter'
    max_workers = 32
    
    concatenate_csv = True # True if outputing single .csv file per simulation (and deleting individual csv)

    for root, dirs, files in os.walk(input_dir, topdown=False):
        for fname in files:
            if '.json' in fname:

                stime = time.time()
                json_fname = os.path.join(root, fname)
                print('starting simulation: {}'.format(json_fname))
                print('starting simulation at: {}'.format(datetime.datetime.fromtimestamp(stime)))

                # init instance of simulation_parameters
                sim_para = simulation_parameters()
                sim_para.read_parameter_json(json_fname)

                # init instance of particles (initializing particles in space)
                particle_list = sim_para.initialize_particles()

                # simulation with multiprocessing
                with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
                    print('workers ready in {}s'.format(time.time() - stime))
                    
                    # molecular crowding
                    if sim_para.simulation_type == 'mc':
                        results = [executor.submit(simulate_particle_mc, sim_para, particle) for particle in particle_list]

                    # interactions with spatially clustered binding sites
                    elif sim_para.simulation_type == 'icbs':
                        results = [executor.submit(simulate_particle_icbs, sim_para, particle) for particle in particle_list]

                    # confining boundary
                    elif sim_para.simulation_type == 'cb':
                        results = [executor.submit(simulate_particle_cb, sim_para, particle) for particle in particle_list]

                    for pp in concurrent.futures.as_completed(results):
                        print(pp.result())
                        print('time elapsed: {}s'.format(time.time() - stime))
                
                print('simulation complete in: {}s'.format(time.time() - stime))
                
                # concatenating data into a final json
                if concatenate_csv:
                    cstart = time.time()
                    print('concatenating csv...')

                    dict_store = {}
                    csv_list = [csv for csv in os.listdir(sim_para.savefile) if csv.startswith('tr_id_')]

                    for csv in csv_list:
                        tr_id = int(os.path.splitext(csv)[0].split('_')[-1])
                        dict_store[tr_id] = pd.read_csv(os.path.join(sim_para.savefile, csv)).to_json()
                        os.remove(os.path.join(sim_para.savefile, csv))
                                                        
                    with open(os.path.join(sim_para.savefile, 'output.json'), 'w') as file:
                        json.dump(dict_store, file, indent=4)

                    print('concatenating complete in: {}s'.format(time.time() - cstart))
                    

if __name__ == '__main__':
    main()