import os 
import concurrent.futures
import trimesh

def main():

    trimesh_synapse_fname = 'bloss18_dataset_1_series_synapse_sv_repaired_hull.stl'
    # load trimesh
    # trimesh_axon = trimesh.load_mesh(self.trimesh_axon_fname)
    trimesh_synapse = trimesh.load_mesh(trimesh_synapse_fname)

    def load_trimesh(trimesh_synapse_fname):
        return trimesh.load_mesh(trimesh_synapse_fname)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(load_trimesh, trimesh_synapse_fname)]

    for pp in concurrent.futures.as_completed(results):
        print(pp.result())

if __name__ == '__main__':
    main()