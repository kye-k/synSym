import json
import os 
os.chdir(os.path.dirname(__file__))

def create_parameter_json(fname, parameter_dictionary):
    with open(fname, 'w') as f:
        json.dump(parameter_dictionary, f, indent="")

