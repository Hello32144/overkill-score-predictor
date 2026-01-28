import json
import numpy as np

def generate_data(output_json, teams, num_matches):
    structure = {"root":{}}
    ranges_for_phases = {
        "autoFuel" : [0,25],
        "transitionFuel" : [0,10],
        "shift1Fuel" : [8,20],
        "shift2Fuel" : [0,0],
        "shift3Fuel" : [8,20],
        "shift4Fuel" : [0,0],
        "endgameFuel" : [8,20],
    }
    generated_matches = {}
    for team_id in teams:
        for i in range(1, num_matches+1):
            match_data = {}
            for phase, limits in ranges_for_phases.items():
                match_data[phase] = int(np.random.randint(limits[0], limits[1]+1))

            generated_matches[f"Match_{i}"] = match_data
            structure["root"][team_id] = generated_matches
        with open (output_json, "w")as f:
            json.dump(structure, f, indent = 4)
generate_data("m.json", ["254", "3464"],  1000)